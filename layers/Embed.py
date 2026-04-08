"""
Embedding layers for time series transformers.

This module provides various embedding strategies for time series data,
including positional embeddings, token embeddings, temporal embeddings,
and specialized patch embeddings for transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.masking import GLMMaskProcessor4TSGPU


class PositionalEmbedding(nn.Module):
    """
    Positional embedding layer for transformer models.
    
    This implementation uses learnable embeddings instead of fixed sinusoidal encodings,
    supporting both single and dual-position encoding schemes for complex masking strategies.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the positional embedding layer.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            max_len (int): Maximum sequence length. Defaults to 5000.
        """
        super(PositionalEmbedding, self).__init__()
        # Use learnable embeddings instead of fixed sinusoidal encodings
        self.position_embedding_1 = nn.Embedding(1024, d_model).float()
        self.position_embedding_2 = nn.Embedding(1024, d_model).float()

    def forward(self, pos_1, pos_2=None):
        """
        Generate positional embeddings for input positions.
        
        Args:
            pos_1 (torch.Tensor): Primary position indices.
            pos_2 (torch.Tensor, optional): Secondary position indices for GLM-style masking.
            
        Returns:
            torch.Tensor: Positional embeddings of shape [B, L, d_model].
        """
        if pos_2 is None:
            return self.position_embedding_1(pos_1)
            
        # Adjust positions to 0-based indexing, with -1 for padding
        pos_1 -= 1
        pos_2 -= 1
        
        # Create padding masks
        padding_mask_1 = pos_1 != -1
        padding_mask_2 = pos_2 != -1
        
        # Set padding positions to 0 for embedding lookup
        valid_positions_1 = torch.where(padding_mask_1, pos_1, torch.tensor(0))
        valid_positions_2 = torch.where(padding_mask_2, pos_2, torch.tensor(0))
        
        # Get embeddings and zero out padding positions
        pos_embed_1 = self.position_embedding_1(valid_positions_1)
        pos_embed_1[~padding_mask_1] = 0
        pos_embed_2 = self.position_embedding_2(valid_positions_2)
        pos_embed_2[~padding_mask_2] = 0
        
        # Combine primary and secondary position embeddings
        return pos_embed_1 + pos_embed_2


class TokenEmbedding(nn.Module):
    """
    Token embedding layer using 1D convolutional projection.
    
    This layer projects input features onto a d-dimensional vector space
    using a 1D convolution with circular padding.
    """
    
    def __init__(self, c_in, d_model):
        """
        Initialize the token embedding layer.
        
        Args:
            c_in (int): Number of input channels/features.
            d_model (int): Dimension of the model embeddings.
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Apply token embedding to input features.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, C].
            
        Returns:
            torch.Tensor: Embedded tensor of shape [B, L, d_model].
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Fixed positional embedding using sinusoidal functions.
    
    This implementation follows the original transformer positional encoding,
    creating fixed sinusoidal embeddings that are not updated during training.
    """
    
    def __init__(self, c_in, d_model):
        """
        Initialize the fixed embedding layer.
        
        Args:
            c_in (int): Number of input categories/vocabulary size.
            d_model (int): Dimension of the model embeddings.
        """
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        Generate fixed embeddings for input indices.
        
        Args:
            x (torch.Tensor): Input indices tensor.
            
        Returns:
            torch.Tensor: Fixed embeddings tensor.
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time-related features.
    
    This layer creates embeddings for various temporal components such as
    month, day, weekday, hour, and minute, allowing the model to capture
    temporal patterns in the data.
    """
    
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        Initialize the temporal embedding layer.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            embed_type (str): Type of embedding ('fixed' or 'learned'). Defaults to 'fixed'.
            freq (str): Frequency of the time series data. Defaults to 'h' (hourly).
        """
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        Generate temporal embeddings for time features.
        
        Args:
            x (torch.Tensor): Time features tensor of shape [B, L, 5] containing
                            [month, day, weekday, hour, minute] information.
                            
        Returns:
            torch.Tensor: Combined temporal embeddings of shape [B, L, d_model].
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Time feature embedding using linear projection.
    
    This layer projects time-related features directly using a linear transformation.
    """
    
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        Initialize the time feature embedding layer.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            embed_type (str): Type of embedding. Defaults to 'timeF'.
            freq (str): Frequency of the time series data. Defaults to 'h' (hourly).
        """
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        Project time features to model dimension.
        
        Args:
            x (torch.Tensor): Time features tensor.
            
        Returns:
            torch.Tensor: Projected time features.
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Comprehensive data embedding combining value, positional, and temporal embeddings.
    
    This layer combines multiple embedding types to create rich representations
    of time series data for transformer models.
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Initialize the data embedding layer.
        
        Args:
            c_in (int): Number of input channels/features.
            d_model (int): Dimension of the model embeddings.
            embed_type (str): Type of temporal embedding. Defaults to 'fixed'.
            freq (str): Frequency of the time series data. Defaults to 'h' (hourly).
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Apply comprehensive data embedding to input features.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, C].
            x_mark (torch.Tensor): Time mark tensor of shape [B, L, D] or None.
            
        Returns:
            torch.Tensor: Embedded tensor of shape [B, L, d_model].
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    Inverted data embedding for multivariate time series.
    
    This layer is designed for inverted transformer architectures where
    variables are treated as sequence elements and time points as features.
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Initialize the inverted data embedding layer.
        
        Args:
            c_in (int): Number of input time points/features.
            d_model (int): Dimension of the model embeddings.
            embed_type (str): Type of embedding. Defaults to 'fixed'.
            freq (str): Frequency of the time series data. Defaults to 'h' (hourly).
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Apply inverted data embedding to input features.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, L].
            x_mark (torch.Tensor): Time mark tensor or None.
            
        Returns:
            torch.Tensor: Embedded tensor of shape [B, C, d_model].
        """
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding without positional encoding.
    
    This layer combines value and temporal embeddings but excludes positional encoding.
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Initialize the data embedding layer without positional encoding.
        
        Args:
            c_in (int): Number of input channels/features.
            d_model (int): Dimension of the model embeddings.
            embed_type (str): Type of temporal embedding. Defaults to 'fixed'.
            freq (str): Frequency of the time series data. Defaults to 'h' (hourly).
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Apply data embedding without positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, C].
            x_mark (torch.Tensor): Time mark tensor or None.
            
        Returns:
            torch.Tensor: Embedded tensor without positional encoding.
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch embedding for time series transformers.
    
    This layer converts time series data into patches and applies embedding,
    supporting both standard and masked patch embedding for pre-training.
    """
    
    def __init__(self, config, d_model, patch_len, stride, padding, dropout):
        """
        Initialize the patch embedding layer.
        
        Args:
            config (object): Configuration object containing task parameters.
            d_model (int): Dimension of the model embeddings.
            patch_len (int): Length of each patch.
            stride (int): Stride for patching.
            padding (int): Padding size for patching.
            dropout (float): Dropout probability.
        """
        super(PatchEmbedding, self).__init__()
        # Patching parameters
        self.config = config
        self.patch_len = patch_len
        
        # Special handling for imputation task
        if config.task_name == 'imputation':
            self.patch_len = config.enc_in
            
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone: Input encoding through linear projection
        self.value_embedding = nn.Linear(self.patch_len, d_model, bias=False)
        
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Learnable embeddings for masking and special tokens
        self.mask_embedding = torch.nn.Parameter(torch.randn(1, patch_len))
        self.Mask = GLMMaskProcessor4TSGPU(mask_ratio=0.3)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.start_token = torch.nn.Parameter(torch.randn(1, patch_len))

    def forward(self, x, mask=False):
        """
        Apply patch embedding to input time series.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, L] or [B, L, C].
            mask (bool): Whether to apply masking for pre-training. Defaults to False.
            
        Returns:
            tuple or torch.Tensor: For masked embedding returns multiple tensors,
                                  for standard embedding returns embedded tensor and n_vars.
        """
        # Determine number of variables based on task
        n_vars = x.shape[1]
        if self.config.task_name == 'imputation':
            n_vars = x.shape[2]
            
        # Apply padding
        x = self.padding_patch_layer(x)
        
        # Perform patching for non-imputation tasks
        if self.config.task_name != 'imputation':
            x_patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # B C N P
            x = torch.reshape(x_patch, (x_patch.shape[0] * x_patch.shape[1], x_patch.shape[2], x_patch.shape[3]))  # B*C N P
            
        # Apply masked embedding for pre-training
        if mask is True:
            data_dict = self.Mask(x)
            position = self.position_embedding(data_dict['position_ids_1'], data_dict['position_ids_2'])
            mask_x = data_dict['input_patches']
            
            # Apply mask embedding to masked positions
            for i in range(len(data_dict['input_patches'])):
                mask_x[i, data_dict['mask_pos'][i]] = self.mask_embedding
                
            # Add start tokens
            mask_x[data_dict['start_token_indices']] = self.start_token
            
            # Apply value and positional embedding
            x = self.value_embedding(mask_x) + position
            atten_mask = data_dict['attention_masks']
            
            return (self.dropout(x), data_dict['input_patches'], atten_mask, 
                    data_dict['mask_ids'], data_dict['end_token_indices'], 
                    data_dict['position_ids_1'], data_dict['position_ids_2'])
        else:
            # Standard embedding without masking
            position_indices = torch.arange(x.size(1)).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
            position_encodings = self.position_embedding(position_indices)
            x = self.value_embedding(x) + position_encodings
            return self.dropout(x), n_vars

    def random_mask(self, sequence, mask_prob=0.3):
        """
        Apply random masking to time series data.
        
        Args:
            sequence (torch.Tensor): Input time series of shape [B, L, D].
            mask_prob (float): Probability of masking each time point. Defaults to 0.3.
            
        Returns:
            tuple: (masked_sequence, mask) where:
                - masked_sequence: Sequence with masked positions zeroed out
                - mask: Binary mask indicating masked positions
        """
        batch_size, seq_length, feature_dim = sequence.shape

        # Initialize mask tensor
        mask = torch.zeros(batch_size, seq_length).cuda()

        # Calculate number of positions to mask
        num_masked = int(mask_prob * seq_length)

        # Randomly select positions to mask for each sample
        for i in range(batch_size):
            mask_indices = torch.randperm(seq_length)[:num_masked]
            mask[i, mask_indices] = 1

        # Expand mask to match sequence dimensions
        mask = mask.unsqueeze(-1).expand(-1, -1, feature_dim)

        # Apply mask to sequence
        masked_sequence = sequence * (1 - mask)

        return masked_sequence, mask
