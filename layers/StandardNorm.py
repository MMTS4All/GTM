"""
Standard normalization layer for time series.

This module implements a flexible normalization layer that can be used
for time series forecasting with various normalization strategies.
"""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Flexible normalization layer for time series data.
    
    This module provides various normalization strategies for time series data,
    including standard normalization, last-value subtraction, and bypass modes.
    """
    
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        Initialize the normalization layer.
        
        Args:
            num_features (int): Number of features or channels.
            eps (float): Value added for numerical stability. Defaults to 1e-5.
            affine (bool): If True, the layer has learnable affine parameters. Defaults to False.
            subtract_last (bool): If True, subtract the last value instead of mean. Defaults to False.
            non_norm (bool): If True, bypass normalization entirely. Defaults to False.
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Apply normalization or denormalization.
        
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
        """Initialize affine parameters if enabled."""
        # initialize affine params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Compute statistics for normalization.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, num_features].
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
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
        if self.non_norm:
            return x
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
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
