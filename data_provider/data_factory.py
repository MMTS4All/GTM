"""
Data provider factory for time series datasets.

This module provides a unified interface for creating data loaders for various
time series datasets, supporting both single-GPU and distributed training scenarios.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .data_loader import (Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute,
                          Dataset_M4, Dataset_elc, Dataset_traffic, Dataset_weather,
                          MSLSegLoader, PSMSegLoader, SMAPSegLoader, SMDSegLoader,
                          SWATSegLoader)
from .utsdataset import UTSDataset

# Constants
DEFAULT_UTSD_SUBSET_NAME = r'UTSD-12G'


# Dictionary mapping dataset names to their respective classes
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'utsd': UTSDataset,
    'weather': Dataset_weather,
    'traffic': Dataset_traffic,
    'elc': Dataset_elc,
    'msl': MSLSegLoader,
    'smap': SMAPSegLoader,
    'swat': SWATSegLoader,
    'psm': PSMSegLoader,
    'smd': SMDSegLoader
}


def data_provider(args, flag, data=None):
    """
    Create dataset and dataloader instances for the specified data split.
    
    This function handles dataset creation and DataLoader configuration, with
    special support for distributed training scenarios. It automatically adjusts
    batch sizes and shuffling behavior based on the training context.
    
    Args:
        args (argparse.Namespace): Configuration arguments containing data settings.
        flag (str): Data split identifier ('train', 'val', 'test', or 'pred').
        data (optional): Pre-loaded data for UTSD dataset. Defaults to None.
        
    Returns:
        tuple: (dataset, dataloader) for the specified data split.
        
    Raises:
        ValueError: If global batch size is not divisible by world size in distributed training.
    """
    # Select appropriate dataset class based on configuration
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # Configure data loader parameters based on split type
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        
    # Create dataset instance with appropriate parameters
    if args.data == 'utsd':
        # Special handling for UTSD dataset
        data_set = Data(data, root_path=args.root_path, subset_name=DEFAULT_UTSD_SUBSET_NAME, 
                       input_len=args.seq_len, output_len=args.pred_len, flag=flag)
    elif args.task_name == 'anomaly_detection':
        # Special handling for anomaly detection tasks
        data_set = Data(
            args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag
        )
    else:
        # Standard dataset creation
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )   
        
    print(flag, len(data_set))
    
    # Check if distributed training is active
    is_distributed = dist.is_available() and dist.is_initialized()
    sampler = None
    shuffle_flag_for_dataloader = shuffle_flag

    # Configure distributed training settings
    if is_distributed:
        # Create DistributedSampler for distributed training
        print(f"Rank {dist.get_rank()}: Creating DistributedSampler for {flag} data.")
        sampler = DistributedSampler(
            data_set,
            shuffle=shuffle_flag,  # Shuffle during training, not during testing
            drop_last=drop_last    # Follow scenario's drop_last logic
        )
        # Disable DataLoader shuffling when using DistributedSampler
        shuffle_flag_for_dataloader = False

        # Calculate per-GPU batch size for distributed training
        world_size = dist.get_world_size()
        # Ensure global batch size is divisible by world size
        if batch_size % world_size != 0:
            raise ValueError(
                f"Global batch size ({batch_size}) must be divisible by world size ({world_size}). "
                f"Please adjust args.batch_size or number of GPUs."
            )
        per_gpu_batch_size = batch_size // world_size
    else:
        # Single GPU/CPU: per-GPU batch size equals global batch size
        per_gpu_batch_size = batch_size

    # Create DataLoader with configured parameters
    data_loader = DataLoader(
        data_set,
        batch_size=per_gpu_batch_size,
        shuffle=shuffle_flag_for_dataloader,
        num_workers=args.num_workers,
        drop_last=drop_last,
        sampler=sampler,
        pin_memory=True  # Enable pinned memory for faster GPU transfer
    )

    return data_set, data_loader
