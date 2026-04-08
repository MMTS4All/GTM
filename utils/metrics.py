"""
Evaluation metrics for time series forecasting.

This module provides various metrics commonly used to evaluate
the performance of time series forecasting models.
"""

import numpy as np


def RSE(pred, true):
    """
    Calculate Relative Squared Error.
    
    RSE measures the ratio of the squared error of the prediction
    to the squared error of a naive predictor (mean of true values).
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        float: Relative Squared Error.
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Calculate Pearson Correlation Coefficient.
    
    CORR measures the linear correlation between predicted and true values.
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        float: Pearson Correlation Coefficient.
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Calculate Mean Absolute Error.
    
    MAE measures the average absolute difference between predicted and true values.
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        float: Mean Absolute Error.
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Calculate Mean Squared Error.
    
    MSE measures the average squared difference between predicted and true values.
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        float: Mean Squared Error.
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Calculate Root Mean Squared Error.
    
    RMSE measures the square root of the average squared difference
    between predicted and true values, in the same units as the data.
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        float: Root Mean Squared Error.
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE measures the average absolute percentage difference
    between predicted and true values.
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        float: Mean Absolute Percentage Error.
    """
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """
    Calculate Mean Squared Percentage Error.
    
    MSPE measures the average squared percentage difference
    between predicted and true values.
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        float: Mean Squared Percentage Error.
    """
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    Calculate multiple evaluation metrics.
    
    This function computes several common metrics for evaluating
    time series forecasting performance.
    
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): True values.
        
    Returns:
        tuple: (mae, mse, rmse, mape, mspe) evaluation metrics.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
