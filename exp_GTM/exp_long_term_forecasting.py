"""
Experiment class for long-term forecasting with the GTM model.

This module implements the training, validation, and testing procedures for
long-term time series forecasting using the GTM (Graph-based Time Series Model).
It includes support for distributed training with DeepSpeed, model checkpointing,
and performance profiling capabilities.
"""

import json
import os
import sys
import time
import types
from collections import defaultdict, OrderedDict
from unittest.mock import patch

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.data_factory import data_provider
from data_provider.utsdataset import prepareUTSD
from models import GTM
from utils.tools import EarlyStopping, adjust_learning_rate

# Constants
DEFAULT_CUDA_DEVICES = "0,1,2,3,4,5,6,7"
DEFAULT_MASTER_PORT = 12355
LOG_INTERVAL = 300
EPSILON = 1e-8

# Mock mpi4py imports to prevent initialization issues
sys.modules['mpi4py'] = types.ModuleType('mpi4py')
sys.modules['mpi4py.MPI'] = None

# Mock mpi_discovery function to prevent MPI initialization
import deepspeed.comm.comm as _comm
_comm.mpi_discovery = lambda *args, **kwargs: None

# Configure environment
os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_CUDA_DEVICES
os.environ['PYTHONWARNINGS'] = 'ignore'


class AFNOProfiler:
    """
    Profiler for measuring computational performance of AFNO layers.
    
    This profiler tracks FFT/IFFT operations and filtering times for AFNO layers,
    providing detailed timing statistics for performance optimization.
    """
    
    def __init__(self, cuda=True):
        """
        Initialize the AFNO profiler.
        
        Args:
            cuda (bool): Whether to use CUDA timing events. Defaults to True.
        """
        self.cuda = cuda and torch.cuda.is_available()
        self.reset()

    def reset(self):
        """Reset all timing statistics."""
        self.fft_time = defaultdict(float)      # Cumulative FFT+IFFT time
        self.filter_time = defaultdict(float)   # Cumulative filter time
        self.cnt = defaultdict(int)             # Call counts

    def enter_layer(self, name):
        """
        Record entry time for a layer.
        
        Args:
            name (str): Layer identifier.
            
        Returns:
            torch.cuda.Event or float: Timing event or timestamp.
        """
        if self.cuda:
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            return evt
        return time.perf_counter()

    def mark_fft_done(self, name):
        """
        Mark completion of FFT operation for timing separation.
        
        Args:
            name (str): Layer identifier.
            
        Returns:
            torch.cuda.Event or None: Timing event for CUDA, None for CPU.
        """
        if self.cuda:
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            return evt
        return None

    def leave_layer(self, name, start_evt, fft_evt=None):
        """
        Record exit time for a layer and accumulate timing statistics.
        
        Args:
            name (str): Layer identifier.
            start_evt: Start timing event/timestamp.
            fft_evt: Optional FFT completion timing event.
        """
        if self.cuda:
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()
            torch.cuda.synchronize()
            total_ms = start_evt.elapsed_time(end_evt)
            if fft_evt is not None:
                fft_ms = start_evt.elapsed_time(fft_evt)
                filter_ms = fft_evt.elapsed_time(end_evt)
            else:
                fft_ms = total_ms
                filter_ms = total_ms
        else:
            total_ms = (time.perf_counter() - start_evt) * 1000
            fft_ms = total_ms
            filter_ms = total_ms

        self.fft_time[name] += fft_ms
        self.filter_time[name] += filter_ms
        self.cnt[name] += 1

    def summary(self, total_samples):
        """
        Print average timing per sample for all profiled layers.
        
        Args:
            total_samples (int): Total number of samples processed.
        """
        print('\n==========  AFNO Average Timing  ==========')
        print('layer_name        fft+ifft(ms/sample)  filter(ms/sample)')
        for name in self.cnt:
            fft_avg = self.fft_time[name] / total_samples
            filter_avg = self.filter_time[name] / total_samples
            print(f'{name:<18}  {fft_avg:8.4f}        {filter_avg:8.4f}')

    def report(self, total_samples):
        """
        Generate detailed timing report as a dictionary.
        
        Args:
            total_samples (int): Total number of samples processed.
            
        Returns:
            dict: Detailed timing statistics for all profiled layers.
        """
        return {name: {
            'fft+ifft_total_ms': self.fft_time[name],
            'fft+ifft_per_sample_ms': self.fft_time[name] / total_samples,
            'filter_total_ms': self.filter_time[name],
            'filter_per_sample_ms': self.filter_time[name] / total_samples,
            'calls': self.cnt[name]}
                for name in self.cnt}


class Exp_Long_Term_Forecast():
    """
    Experiment class for long-term time series forecasting.
    
    This class handles the complete training pipeline for long-term forecasting,
    including data loading, model initialization, distributed training with DeepSpeed,
    validation with early stopping, and testing with performance evaluation.
    """

    def __init__(self, args):
        """
        Initialize the long-term forecasting experiment.
        
        Args:
            args (argparse.Namespace): Configuration arguments for the experiment.
        """
        self.args = args
        self.device = self._acquire_device()
        self.model_dict = {
            'GTM': GTM.Model,
        }
        self.model = self._build_model()
        
        # Load pre-trained model weights if not using UTSD dataset
        if self.args.data != 'utsd':
            model_parameters = torch.load(self.args.pretrain_model_path)
            new_state_dict = OrderedDict()
            # Remove 'module.' prefix from parameter names
            for key, value in model_parameters.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict, strict=True)
            
        # Prepare UTSD dataset if specified
        if args.data == 'utsd':
            self.data = prepareUTSD(root_path='/data/dataset/train', subset_name=r'UTSD-12G', flag='train',
                                    input_len=self.args.seq_len, output_len=self.args.pred_len).load_data()
                                    
        # Initialize optimizer and DeepSpeed distributed training
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self._select_optimizer()
        with open('../configs/ds_config.json') as f:
            ds_config = json.load(f)
        
        # Mock distributed training when not actually needed
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', str(DEFAULT_MASTER_PORT))
        
        # self.model, self.optimizer, _, _ = deepspeed.initialize(
        #     args=self.args,
        #     config=ds_config,
        #     model=self.model,
        #     optimizer=self.optimizer,
        #     model_parameters=parameters
        # )


    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        """
        Build and return the model instance.
        
        Returns:
            nn.Module: Initialized model instance.
        """
        model = self.model_dict[self.args.model](self.args)
        return model.to(self.device)

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
            Adam: Configured optimizer instance.
        """
        # Use standard Adam optimizer
        model_optim = Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=EPSILON,
            weight_decay=self.args.weight_decay
        )
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
                batch_x = batch_x.float().to(self.model.device)
                batch_y = batch_y.float().to(self.model.device)
                
                # Forward pass through model
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model(batch_x, time_gra)
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model(batch_x, time_gra)
                    batch_y = batch_y.to(self.model.device)
                    
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
            deepspeed.DeepSpeedEngine: Trained model.
        """
        # Prepare data loaders
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        criterion = self._select_criterion()
        
        # Create checkpoint directory on main process
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
                
            # Batch training loop
            for i, (batch_x, batch_y, time_gra) in enumerate(train_loader):
                time_now = time.time()
                iter_count += 1
                self.optimizer.zero_grad()
                
                batch_x = batch_x.float().to(self.model.device)
                batch_y = batch_y.float().to(self.model.device)

                # Forward pass
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model(batch_x, time_gra)
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model(batch_x, time_gra)
                    batch_y = batch_y.to(self.model.device)
                    
                loss = criterion(outputs, batch_y)
                
                # Distributed loss reduction
                cur_device = torch.cuda.current_device()
                loss_tensor = torch.tensor(loss.item(), device=f'cuda:{cur_device}')
                if dist.is_initialized():
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_tensor /= dist.get_world_size()
                train_loss.append(loss_tensor.item())
                
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
                    # for ds framework
                    # self.model_parallel.backward(loss)
                    # self.model_parallel.step()
                    loss.backward()
                    self.optimizer.step()

                    
            # Validation and logging on main process
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            # Early stopping check
            if early_stopping.early_stop:
                self.model._stop_training = True
                    
            # Check for early stopping signal
            if hasattr(self.model, '_stop_training') and self.model._stop_training:
                break
                
            # Adjust learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        # Load best model checkpoint
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
        # Prepare test data
        test_data, test_loader = self._get_data(flag='test')
        
        # Load pre-trained model if specified
        if test:
            print('loading models')
            model_parameters = torch.load('')
            new_state_dict = OrderedDict()
            # Add 'module.' prefix for compatibility
            for key, value in model_parameters.items():
                new_key = 'module.' + key
                new_state_dict[new_key] = value.to(self.args.device)
            self.model.load_state_dict(new_state_dict)

        # Initialize result storage
        lb_windows = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        total_samples = 0
        sample_cost_time = []
        
        # Testing loop
        with torch.no_grad():
            for i, (batch_x, batch_y, time_gra) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.model.device)
                batch_y = batch_y.float().to(self.model.device)

                # Forward pass
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model(batch_x, time_gra)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = patch_x
                else:
                    time_now = time.time()
                    outputs = self.model(batch_x, time_gra)
                    sample_cost_time.append(time.time() - time_now)
                    outputs = outputs.detach().cpu().numpy()
                y = batch_y.detach().cpu().numpy()
                
                # Accumulate results
                if i == 0:
                    lb_windows = batch_x.detach().cpu().numpy()
                    preds = outputs
                    trues = y
                else:
                    lb_window = batch_x.detach().cpu().numpy()
                    lb_windows = np.concatenate((lb_windows, lb_window), axis=0)
                    preds = np.concatenate((preds, outputs), axis=0)
                    trues = np.concatenate((trues, y), axis=0)
                total_samples += batch_x.size(0)

        # Calculate and print test metrics
        print('test shape:', preds.shape, trues.shape)
        mse = F.mse_loss(torch.from_numpy(preds), torch.from_numpy(trues))
        mae = F.l1_loss(torch.from_numpy(preds), torch.from_numpy(trues))
        print('test shape:', preds.shape, trues.shape)
        print(f'the test loss {mse, mae}')
        return