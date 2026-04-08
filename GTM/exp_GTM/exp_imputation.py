import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import torch
import matplotlib.pyplot as plt
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


class Exp_Imputation():
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
                                                                        config=ds_config,distributed_port=29503)
        # if self.args.data != 'utsd':
        #     model_parameters = torch.load(os.path.join(
        #         '/home/dmz-ai/lxj/GTM/checkpoints/pre_train_etth1_utsd_GTM_Monthly_ftM_idFalse_sl1440_pl0_dm768_df32_lr1e-05_bs1024/checkpoint.pth'))
        #     new_state_dict = OrderedDict()
        #     # 遍历当前的模型参数并修改键
        #     for key, value in model_parameters.items():
        #         # 去掉参数名称前的所有 'module.module.'
        #         # if 'head.weight' in new_state_dict:
        #         #     del new_state_dict['head.weight']
        #         # if 'head.bias' in new_state_dict:
        #         #     del new_state_dict['head.bias']
        #         new_key = 'module.'+key
        #         if new_key not in ['module.module.patch_embedding.mask_embedding','module.module.patch_embedding.start_token','module.module.patch_embedding.value_embedding.weight','module.module.head.weight','module.module.head.bias']:
        #             new_state_dict[new_key] = value.to(self.args.device)
        #         # new_state_dict[key] = value.to(self.args.device)
        #     self.model_parallel.load_state_dict(new_state_dict, strict=False)
        #     # for name, param in self.model_parallel.named_parameters():
        #     #     if param.requires_grad:
        #     #         print(name, param.data.shape)

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
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.model_parallel.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                # decoder input
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x, time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(inp, time_gra,mask)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])

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
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.model_parallel.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                # decoder input
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x,time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(inp,time_gra,mask)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
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
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
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
            model_parameters = torch.load('')

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



        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,time_gra) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.model_parallel.device)
                batch_y = batch_y.float().to(self.model_parallel.device)
                # encoder - decoder
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.model_parallel.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                # decoder input
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x, time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                else:
                    outputs = self.model_parallel(inp, time_gra,mask)
                    outputs = outputs.detach().cpu().numpy()
                y = batch_x.detach().cpu().numpy()
                if i == 0:
                    preds = outputs
                    trues = y
                    masks = mask.detach().cpu().numpy()
                else:
                    preds = np.concatenate((preds, outputs), axis=0)
                    trues = np.concatenate((trues, y), axis=0)
                    masks = np.concatenate((masks,mask.detach().cpu().numpy()), axis=0)
        print('test shape:', preds.shape, trues.shape)

        mse = F.mse_loss(torch.from_numpy(preds[masks == 0]), torch.from_numpy(trues[masks == 0]))
        mae = F.l1_loss(torch.from_numpy(preds[masks == 0]), torch.from_numpy(trues[masks == 0]))
        print('test shape:', preds.shape, trues.shape)
        print(f'the test loss {mse, mae}')
        prediction_visualization(masks,preds,trues)
        return

def prediction_visualization(masks, preds, trues):
    # lb_windows = lb_windows.swapaxes(1,2)
    # lb_windows = lb_windows.reshape(-1,lb_windows.shape[-1])
    # preds = preds.swapaxes(1,2)
    # preds = preds.reshape(-1,preds.shape[-1])
    # trues = trues.swapaxes(1,2)
    # trues = trues.reshape(-1,trues.shape[-1])
    np.save('/data/lxj_results/GTM_imputation_result/ETTm1/masks.npy', masks)
    np.save('/data/lxj_results/GTM_imputation_result/ETTm1/preds.npy', preds)
    np.save('/data/lxj_results/GTM_imputation_result/ETTm1/trues.npy', trues)
    # lb_windows = lb_windows[:, :, -1]
    # preds = preds[:, :, -1]
    # trues = trues[:, :, -1]
    mse = np.mean((preds[masks==0] - trues[masks==0]) ** 2, axis=1)
    min_error_index = np.argmin(mse)
    print('min_error_index:', min_error_index)
    dim = [0, 430, 861]
    for i in range(preds.shape[-1]):
        mask = masks[:, :, i]
        pred = preds[:, :, i]
        true = trues[:, :, i]
        best_pred = pred[min_error_index]
        best_true = true[min_error_index]
        # Step 2: 可视化结果
        # 假设我们可视化的是序列的某个维度（例如第0维度）变化
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(best_true.shape[0]), best_true, label='GroundTruth', color='blue')  # 绘制真实序列
        plt.plot(np.arange(best_pred.shape[0]), best_pred[mask==0], label='Prediction', color='red',
                 linestyle='--')  # 绘制预测序列
        plt.title(f'electricity dim{i + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
