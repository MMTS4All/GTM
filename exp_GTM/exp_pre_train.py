"""
Experiment class for pre-training with the GTM model.

This module implements the training, validation, and testing procedures for
pre-training the GTM (Graph-based Time Series Model) on time series data.
It includes support for distributed training with DeepSpeed and model checkpointing.
"""

import json
import os
import sys
import time
import warnings
from collections import OrderedDict

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_factory import data_provider
from data_provider.utsdataset import prepareUTSD
from models import GTM
from utils.tools import EarlyStopping, adjust_learning_rate

# Constants
DEFAULT_CUDA_DEVICES = "0,1,2,3,4,5,6,7"
DEFAULT_DEEPSPEED_PORT = 29503
LOG_INTERVAL = 100

# Configure environment
os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_DEVICES
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast():
    """
    Experiment class for pre-training time series model.
    
    This class handles the complete training pipeline for pre-training,
    including data loading, model initialization, distributed training with DeepSpeed,
    validation with early stopping, and testing with performance evaluation.
    """
    
    def __init__(self, args):
        """
        Initialize the pre-training experiment.
        
        Args:
            args (argparse.Namespace): Configuration arguments for the experiment.
        """
        self.args = args
        self.model_dict = {
            'GTM': GTM.Model,
        }
        self.model = self._build_model()
        
        # Prepare UTSD dataset if specified
        if args.data == 'utsd':
            self.data = prepareUTSD(root_path='/data/dataset/train', subset_name=r'UTSD-12G', flag='train',
                                    input_len=self.args.seq_len, output_len=self.args.pred_len).load_data()
                                    
        # Initialize optimizer and DeepSpeed distributed training
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self._select_optimizer()
        with open('configs/ds_config.json') as f:
            ds_config = json.load(f)
            
        self.model_parallel, self.optimizer, _, _ = deepspeed.initialize(
            args=self.args,
            model=self.model,
            optimizer=self.optimizer,
            model_parameters=parameters,
            config=ds_config,
            distributed_port=DEFAULT_DEEPSPEED_PORT
        )
        
        # Load pre-trained model weights if specified
        if self.args.data != 'utsd':
            model_parameters = torch.load(self.args.pretrain_model_path)
            new_state_dict = OrderedDict()
            # Add 'module.' prefix for compatibility
            for key, value in model_parameters.items():
                new_key = 'module.' + key
                new_state_dict[new_key] = value.to(self.args.device)
            self.model_parallel.load_state_dict(new_state_dict, strict=True)

    def _build_model(self):
        """
        Build and return the model instance.
        
        Returns:
            nn.Module: Initialized model instance.
        """
        model = self.model_dict[self.args.model](self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.args.device)

    def _get_data(self, flag):
        """
        Get dataset and dataloader for the specified data split.
        
        Args:
            flag (str): Data split identifier ('train', 'val', 'test').
            
        Returns:
            tuple: (dataset, dataloader) for the specified split.
        """
        if self.args.data == 'utsd':
            data_set, data_loader = data_provider(self.args, flag, self.data)
        else:
            data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Select and configure the optimizer.
        
        Returns:
            optim.Adam: Configured optimizer instance.
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        Select the loss function.
        
        Returns:
            nn.Module: MSE loss function.
        """
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        Validate the model on validation data.
        
        Args:
            vali_data: Validation dataset.
            vali_loader: Validation dataloader.
            criterion: Loss function.
            
        Returns:
            float: Average validation loss.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, time_gra) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.model_parallel.device)
                batch_y = batch_y.float().to(self.model_parallel.device)
                
                # Forward pass through model
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x, time_gra)
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model(batch_x, time_gra)
                    outputs = outputs
                    batch_y = batch_y.to(self.model_parallel.device)
                    
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        Train the model with validation and early stopping.
        
        Args:
            setting (str): Experiment setting identifier.
            
        Returns:
            nn.Module: Trained model.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
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
                batch_y = batch_y.float().to(self.model_parallel.device)
                
                # Forward pass
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x, time_gra)
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(batch_x, time_gra)
                    outputs = outputs
                    batch_y = batch_y.to(self.model_parallel.device)
                    
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                # Log training progress
                if (i + 1) % LOG_INTERVAL == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0

                # Backward pass
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.model_parallel.backward(loss)
                    self.model_parallel.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
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
        """
        Test the trained model and evaluate performance.
        
        Args:
            setting (str): Experiment setting identifier.
            test (int): Flag to indicate if loading pre-trained model. Defaults to 0.
            
        Returns:
            None
        """
        test_data, test_loader = self._get_data(flag='test')
        
        # Load pre-trained model if specified
        if test:
            print('loading models')
            model_parameters = torch.load(os.path.join(''))
            new_state_dict = OrderedDict()
            # Add 'module.' prefix for compatibility
            for key, value in model_parameters.items():
                new_key = 'module.' + key
                new_state_dict[new_key] = value.to(self.args.device)
            self.model_parallel.load_state_dict(new_state_dict)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, time_gra) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.model_parallel.device)
                batch_y = batch_y.float().to(self.model_parallel.device)
                
                # Forward pass
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x, pos_1, pos_2 = self.model_parallel(batch_x, time_gra)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(batch_x, time_gra)
                    outputs = outputs.detach().cpu().numpy()
                y = batch_x.detach().cpu().numpy()
                pos_1 = pos_1.detach().cpu().numpy()
                pos_2 = pos_2.detach().cpu().numpy()
                
                # Accumulate results
                if i == 0:
                    preds = outputs
                    trues = y
                    pos_1_list = pos_1
                    pos_2_list = pos_2
                else:
                    preds = np.concatenate((preds, outputs), axis=0)
                    trues = np.concatenate((trues, y), axis=0)
                    pos_1_list = np.concatenate((pos_1_list, pos_1), axis=0)
                    pos_2_list = np.concatenate((pos_2_list, pos_2), axis=0)
                    
        print('test shape:', preds.shape, trues.shape)
        mse = F.mse_loss(torch.from_numpy(preds), torch.from_numpy(trues))
        mae = F.l1_loss(torch.from_numpy(preds), torch.from_numpy(trues))
        print('test shape:', preds.shape, trues.shape)
        print(f'the test loss {mse, mae}')
        return