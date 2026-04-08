"""
Reversible Instance Normalization (RevIN) for time series.

This module implements RevIN, a normalization technique specifically designed
for time series forecasting that can be reversed after processing.
Paper: https://arxiv.org/abs/2107.05173
Code from: https://github.com/ts-kim/RevIN, with minor modifications
"""

import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series forecasting.
    
    This module normalizes time series data using instance normalization
    that can be reversed after processing, preserving the original scale
    of the data while allowing for more stable training.
    """
    
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        Initialize the RevIN module.
        
        Args:
            num_features (int): Number of features or channels.
            eps (float): Value added for numerical stability. Defaults to 1e-5.
            affine (bool): If True, RevIN has learnable affine parameters. Defaults to True.
            subtract_last (bool): If True, subtract the last value instead of mean. Defaults to False.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        """
        Apply RevIN normalization or denormalization.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, num_features].
            mode (str): Operation mode, either 'norm' or 'denorm'.
            
        Returns:
            Tensor: Normalized or denormalized tensor of the same shape.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        """Initialize RevIN affine parameters."""
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Compute statistics for normalization.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, num_features].
        """
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        Apply normalization to input tensor.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Normalized tensor.
        """
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        Apply denormalization to input tensor.
        
        Args:
            x (Tensor): Normalized tensor.
            
        Returns:
            Tensor: Denormalized tensor in original scale.
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
