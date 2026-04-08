"""
Main entry point for pre-training experiments.

This script serves as the primary interface for running pre-training
experiments with the GTM model. It handles configuration loading, environment setup,
and experiment execution for unsupervised pre-training on time series data.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from exp_GTM.exp_pre_train import Exp_Long_Term_Forecast


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    """
    Main execution block for running pre-training experiments.
    
    This section initializes random seeds for reproducibility, loads configuration,
    sets up the experimental environment, and executes pre-training
    procedures for the GTM model.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of scripts directory)
    project_root = os.path.dirname(script_dir)
    # Default config path relative to project root
    default_config = os.path.join(project_root, 'configs', 'config.yaml')
    
    # Define argument parser with minimal arguments
    parser = argparse.ArgumentParser(description='GTM Pre-training')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to configuration file')
    parser.add_argument('--section', type=str, default='pretrain',
                        help='Section name in config file (pretrain/experiment)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8', errors='ignore') as f:
        config_data = yaml.safe_load(f)
        section_config = config_data.get(args.section, {})
        
        # Set configuration values to args object
        for key, value in section_config.items():
            setattr(args, key, value)
    
    # Set random seed for reproducibility
    if hasattr(args, 'random_seed') and args.random_seed is not None:
        set_seed(args.random_seed)
    
    # Configure CUDA devices and deepspeed settings for pre-training
    os.environ["CUDA_VISIBLE_DEVICES"] = getattr(args, 'cuda_devices', "0,1,2,3,4,5,6,7")
    deepspeed_port = getattr(args, 'deepspeed_port', 29501)
    os.environ['DEEPSPEED_CONFIG'] = f'{{"ports":{{"master_port":{deepspeed_port}}}}}'
    
    # Configure GPU usage
    args.use_gpu = True if torch.cuda.is_available() and getattr(args, 'use_gpu', True) else False

    # Configure multi-GPU settings
    if args.use_gpu and getattr(args, 'use_multi_gpu', True):
        args.devices = getattr(args, 'devices', '0').replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        args.gpu = getattr(args, 'gpu', 0)
    
    # Set device
    args.device = torch.device('cuda:{}'.format(args.gpu))
    
    print('Args in experiment:')
    print(args)

    # Execute training or testing based on configuration
    if args.is_training:
        # Initialize experiment for pre-training
        Exp = Exp_Long_Term_Forecast(args)
        
        # Create experiment setting identifier
        setting = '{}_{}_{}_{}_{}_ft{}_id{}_sl{}_pl{}_dm{}_df{}_lr{}_bs{}'.format(
            args.task_name,
            args.model_id,
            args.data,
            args.model,
            getattr(args, 'seasonal_patterns', ''),
            args.features,
            args.individual,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.learning_rate,
            args.batch_size)
            
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        Exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        Exp.test(setting)
        torch.cuda.empty_cache()
    else:
        # Initialize experiment for testing
        Exp = Exp_Long_Term_Forecast(args)
        
        # Create experiment setting identifier
        setting = '{}_{}_{}_{}_{}_ft{}_id{}_sl{}_pl{}_dm{}_df{}_lr{}_bs{}'.format(
            args.task_name,
            args.model_id,
            args.data,
            args.model,
            getattr(args, 'seasonal_patterns', ''),
            args.features,
            args.individual,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.learning_rate,
            args.batch_size)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        Exp.test(setting, test=1)
        torch.cuda.empty_cache()