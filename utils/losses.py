# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS models (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS models and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for time series forecasting models.

This module provides specialized loss functions commonly used in
time series forecasting, including MAPE, sMAPE, and MASE variants
that handle masked data appropriately.
"""

import torch as t
import torch.nn as nn
import numpy as np


def divide_no_nan(a, b):
    """
    Safe division that replaces NaN or Inf results with 0.
    
    This function performs element-wise division and handles
    division by zero by replacing undefined results with 0.
    
    Args:
        a (torch.Tensor): Numerator tensor.
        b (torch.Tensor): Denominator tensor.
        
    Returns:
        torch.Tensor: Result of a/b with NaN and Inf replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE) loss function.
    
    This loss function calculates the mean absolute percentage error
    between forecast and target values, with proper handling of
    masked data and division by zero.
    """
    
    def __init__(self):
        """Initialize the MAPE loss module."""
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        Compute MAPE loss.
        
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
        
        Args:
            insample (torch.Tensor): Insample values. Shape: batch, time_i
            freq (int): Frequency value.
            forecast (torch.Tensor): Forecast values. Shape: batch, time
            target (torch.Tensor): Target values. Shape: batch, time
            mask (torch.Tensor): 0/1 mask. Shape: batch, time
            
        Returns:
            torch.float: MAPE loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) loss function.
    
    This loss function calculates the symmetric mean absolute percentage error
    between forecast and target values, providing a bounded percentage error metric.
    """
    
    def __init__(self):
        """Initialize the sMAPE loss module."""
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        Compute sMAPE loss.
        
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)
        
        Args:
            insample (torch.Tensor): Insample values. Shape: batch, time_i
            freq (int): Frequency value.
            forecast (torch.Tensor): Forecast values. Shape: batch, time
            target (torch.Tensor): Target values. Shape: batch, time
            mask (torch.Tensor): 0/1 mask. Shape: batch, time
            
        Returns:
            torch.float: sMAPE loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    """
    Mean Absolute Scaled Error (MASE) loss function.
    
    This loss function calculates the mean absolute scaled error,
    which scales the forecast error by the in-sample mean absolute error.
    """
    
    def __init__(self):
        """Initialize the MASE loss module."""
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        Compute MASE loss.
        
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf
        
        Args:
            insample (torch.Tensor): Insample values. Shape: batch, time_i
            freq (int): Frequency value.
            forecast (torch.Tensor): Forecast values. Shape: batch, time_o
            target (torch.Tensor): Target values. Shape: batch, time_o
            mask (torch.Tensor): 0/1 mask. Shape: batch, time_o
            
        Returns:
            torch.float: MASE loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
