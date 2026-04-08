"""
Main entry point for time series forecasting experiments.

This script serves as the primary interface for running long-term forecasting
experiments with the GTM model. It handles configuration loading, environment setup,
and experiment execution for various datasets and configurations.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from exp_GTM.exp_long_term_forecasting import Exp_Long_Term_Forecast


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


def get_pred_len_to_seq_len_mapping():
    """
    Get mapping between prediction length and sequence length.
    
    Returns:
        dict: Mapping dictionary.
    """
    return {
        96: 672,
        192: 672,
        336: 1440,
        720: 1440
    }


if __name__ == '__main__':
    """
    Main execution block for running forecasting experiments.
    
    This section loads configuration, sets up the experimental
    environment, and executes training/testing procedures for the GTM model.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root directory (parent of scripts directory)
    project_root = os.path.dirname(script_dir)
    # Default config path relative to project root
    default_config = os.path.join(project_root, 'configs', 'config.yaml')
    
    # Define argument parser with minimal arguments
    parser = argparse.ArgumentParser(description='GTM Forecasting')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to configuration file')
    parser.add_argument('--section', type=str, default='experiment',
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
    
    # Convert pretrain_model_path to absolute path if it's relative
    if hasattr(args, 'pretrain_model_path') and args.pretrain_model_path:
        if not os.path.isabs(args.pretrain_model_path):
            args.pretrain_model_path = os.path.join(project_root, args.pretrain_model_path)
            args.pretrain_model_path = os.path.normpath(args.pretrain_model_path)
    
    # Set random seed for reproducibility
    if hasattr(args, 'random_seed') and args.random_seed is not None:
        set_seed(args.random_seed)
    
    # Configure CUDA devices and deepspeed settings
    os.environ["CUDA_VISIBLE_DEVICES"] = getattr(args, 'cuda_devices', "0,1,2,3,4,5,6,7")
    deepspeed_port = getattr(args, 'deepspeed_port', 29503)
    os.environ['DEEPSPEED_CONFIG'] = f'{{"ports":{{"master_port":{deepspeed_port}}}}}'
    
    # Configure GPU usage
    args.use_gpu = True if torch.cuda.is_available() and getattr(args, 'use_gpu', True) else False

    # Configure multi-GPU settings
    if args.use_gpu and getattr(args, 'use_multi_gpu', True):
        args.devices = getattr(args, 'devices', '0,1').replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        args.gpu = getattr(args, 'gpu', 0)
    
    # Set device
    args.device = torch.device('cuda:{}'.format(args.gpu))

    args.enc_in = args.dataset2enc_in.get(args.data)

    print('Args in experiment:')
    print(args)

    # Get prediction length to sequence length mapping
    pred_len_to_seq_len = get_pred_len_to_seq_len_mapping()

    # Execute training or testing based on configuration
    if args.is_training:
        # Training loop for different prediction lengths
        test_pred_lens = [getattr(args, 'test_pred_len', 720)]
        test_iterations = getattr(args, 'test_iterations', 5)
        
        for args.pred_len in test_pred_lens:
            for iteration in range(test_iterations):
                # Adjust sequence length based on prediction length
                if args.pred_len in pred_len_to_seq_len:
                    args.seq_len = pred_len_to_seq_len[args.pred_len]
                if 'pre_train' in args.task_name:
                    args.pred_len = 0
                    
                # Initialize experiment
                Exp = Exp_Long_Term_Forecast(args)

                # Create experiment setting identifier
                setting = '{}_{}_{}_{}_{}_ft{}_id{}_sl{}_pl{}_dm{}_df{}_lr{}_bs{}'.format(
                    args.task_name,
                    args.model_id,
                    args.data,
                    args.model,
                    getattr(args, 'seasonal_patterns', 'Monthly'),
                    args.features,
                    args.individual,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.d_ff,
                    args.learning_rate,
                    args.batch_size)
                    
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                Exp.train(setting)
                Exp.test(setting)
                continue
                
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                Exp.train(setting)

                torch.cuda.empty_cache()
    else:
        # Testing loop for different granularity levels
        test_num_grans = [getattr(args, 'test_num_gran', 5)]
        test_pred_lens = [getattr(args, 'test_pred_len', 96)]
        
        for args.num_gran in test_num_grans:
            for args.pred_len in test_pred_lens:
                Exp = Exp_Long_Term_Forecast(args)
                
                # Create experiment setting identifier
                setting = '{}_{}_{}_{}_{}_ft{}_id{}_sl{}_pl{}_dm{}_df{}_lr{}_bs{}'.format(
                    args.task_name,
                    args.model_id,
                    args.data,
                    args.model,
                    getattr(args, 'seasonal_patterns', 'Monthly'),
                    args.features,
                    args.individual,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.d_ff,
                    args.learning_rate,
                    args.batch_size)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                Exp.test(setting, test=0)
                torch.cuda.empty_cache()
