import sys
sys.path.append('/home/dmz-ai/lxj')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ['PYTHONWARNINGS'] = 'ignore'
import time
import warnings
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dtw_metric import dtw,accelerated_dtw
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.utsdataset import prepareUTSD
import deepspeed
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import GTM
from collections import OrderedDict
# from utils.augmentation import run_augmentation,run_augmentation_single
os.environ['MASTER_PORT'] = '29501'

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection():
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'GTM': GTM.Model,
        }
        self.model =self._build_model()
        if args.data == 'utsd':
            self.data = prepareUTSD(root_path='/data/dataset/train', subset_name=r'UTSD-12G', flag='train',
                                    input_len=self.args.seq_len, output_len=self.args.pred_len).load_data()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self._select_optimizer()
        with open('ds_config.json') as f:
            ds_config = json.load(f)
        self.model_parallel, self.optimizer, _, _ = deepspeed.initialize(args=self.args,model=self.model, optimizer=self.optimizer,model_parameters=parameters,
                                                                        config=ds_config,distributed_port=29505)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.args.device)

    def _get_data(self, flag):
        if self.args.data =='utsd':
            data_set, data_loader = data_provider(self.args, flag, self.data)
        else:
            data_set, data_loader = data_provider(self.args,flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,time_gra) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.model_parallel.device)
                # encoder - decoder
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x,time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(batch_x,time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    loss = criterion(outputs, batch_x)

                total_loss.append(loss.cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)



        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, time_gra) in enumerate(train_loader):
                time_now = time.time()
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.model_parallel.device)
                # decoder input
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x,time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(batch_x,time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    # time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.model_parallel.backward(loss)
                    self.model_parallel.step()
                    # loss.backward()
                    # self.optimizer.step()
                    # for name, param in self.models.named_parameters():
                    #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    #         print(f"NaN or Inf in gradients of {name}")

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            # self.test(setting)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        mse_loss = nn.MSELoss(reduction='none')
        if test:
            print('loading models')
            model_parameters = torch.load(os.path.join('/home/dmz-ai/lxj/FNOformer/checkpoints/anomaly_detection_SMAP_1_smap_FNOformer_Monthly_ftM_idFalse_sl672_pl0_dm768_df32_lr0.0001_bs128/checkpoint.pth'))

            new_state_dict = OrderedDict()
            # 遍历当前的模型参数并修改键
            for key, value in model_parameters.items():
                # 所有参数名称前加上 'module.module.'
                new_key = 'module.'+key
                new_state_dict[new_key] = value.to(self.args.device)
            self.model_parallel.load_state_dict(new_state_dict)

        # for key, value in self.model.named_parameters():
        #     # 去掉参数名称前的所有 'module.module.'
        #
        #     # 打印参数名称和其所在的 GPU
        #     print(f'Parameter: {key}, GPU: {value.get_device()}')  # 打印参数所在的 GPU


        labels = []
        score = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,time_gra) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.model_parallel.device)
                batch_y = batch_y.float().to(self.model_parallel.device)
                # encoder - decoder
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x,time_gra)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(batch_x, time_gra)
                    loss = torch.mean(mse_loss(batch_x, outputs), dim=-1)
                    outputs = outputs.detach().cpu().numpy()
                x = batch_x.detach().cpu().numpy()
                y = batch_y.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()
                if i == 0:
                    score = loss
                    labels = y
                    trues = x
                else:
                    score = np.concatenate((score, loss), axis=0)
                    labels = np.concatenate((labels, y), axis=0)
                    trues = np.concatenate((trues, x), axis=0)
        print('test shape:', labels.shape, score.shape)
        score = np.concatenate(score, axis=0).reshape(-1, 1)
        labels = np.concatenate(labels,axis=0).reshape(-1,1)
        thre = [(90 + (i / 10)) for i in range(100)]
        thres = np.percentile(score, thre)
        F1 = 0
        for i in thres:
            thresh = i
            pred = (score > thresh).astype(int)
            gt = labels.astype(int)
            anomaly_state = False
            # 遍历gt，gt为测试集标签ds
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)
            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                                  average='binary')
            if f_score > F1:
                Accuracy = accuracy
                F1 = f_score
                Precision = precision
                Recall = recall
                thr = thresh
        print("best-thresh:", thr)
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                Accuracy, Precision,
                Recall, F1))

        np.save('/data/lxj_results/GTM_anomaly_result/smap/trues.npy',trues)
        np.save('/data/lxj_results/GTM_anomaly_result/smap/labels.npy',labels)
        np.save('/data/lxj_results/GTM_anomaly_result/smap/scores.npy',score)
        return
