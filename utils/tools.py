"""
Utility functions for time series modeling and training.

This module provides essential utilities for learning rate scheduling,
early stopping, data scaling, visualization, and accuracy calculation.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Constants for learning rate adjustment
LR_DECAY_FACTOR = 0.5
LR_SCHEDULE = {
    2: 5e-5,
    4: 1e-5,
    6: 5e-6,
    8: 1e-6,
    10: 5e-7,
    15: 1e-7,
    20: 5e-8
}


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust learning rate based on epoch and configuration.
    
    This function implements various learning rate scheduling strategies
    including step decay, predefined schedules, and cosine annealing.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be adjusted.
        epoch (int): Current epoch number.
        args (argparse.Namespace): Configuration arguments containing learning rate settings.
            Expected attributes:
            - lradj (str): Learning rate adjustment strategy ('type1', 'type2', 'cosine').
            - learning_rate (float): Base learning rate.
            - train_epochs (int): Total number of training epochs (for cosine scheduling).
    """
    # Configure learning rate adjustment based on strategy
    if args.lradj == 'type1':
        # Step decay: reduce learning rate by half every epoch
        lr_adjust = {epoch: args.learning_rate * (LR_DECAY_FACTOR ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # Predefined schedule with specific learning rates at specific epochs
        lr_adjust = LR_SCHEDULE
    elif args.lradj == "cosine":
        # Cosine annealing schedule
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    
    # Apply learning rate adjustment if scheduled for current epoch
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting during training.
    
    This class monitors validation loss and stops training when the loss
    stops improving for a specified number of consecutive epochs.
    """
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize early stopping mechanism.
        
        Args:
            patience (int): Number of epochs to wait after last improvement.
            verbose (bool): Whether to print verbose messages.
            delta (float): Minimum change in validation loss to qualify as improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Check if training should be stopped based on validation loss.
        
        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Model to save if improved.
            path (str): Path to save the best model checkpoint.
        """
        # Use negative loss as score (higher is better)
        score = -val_loss
        
        if self.best_score is None:
            # First epoch: save initial model
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # No significant improvement: increment counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement: save model and reset counter
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save model checkpoint when validation loss improves.
        
        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Model to save.
            path (str): Directory path to save the checkpoint.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    Dictionary that supports dot notation access to attributes.
    
    This class extends the standard dictionary to allow accessing
    dictionary keys using dot notation (e.g., dict.key instead of dict['key']).
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    Standard scaler for data normalization and denormalization.
    
    This class applies z-score normalization (subtract mean, divide by standard deviation)
    and can reverse the transformation to recover original scale.
    """
    
    def __init__(self, mean, std):
        """
        Initialize the scaler with mean and standard deviation.
        
        Args:
            mean (float or array-like): Mean value(s) for normalization.
            std (float or array-like): Standard deviation value(s) for normalization.
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Apply standard scaling to data.
        
        Args:
            data (array-like): Input data to normalize.
            
        Returns:
            array-like: Normalized data.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverse standard scaling to recover original data scale.
        
        Args:
            data (array-like): Normalized data to denormalize.
            
        Returns:
            array-like: Denormalized data in original scale.
        """
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Visualize ground truth and predicted values.
    
    This function creates a plot comparing actual values with predictions
    and saves it to a PDF file.
    
    Args:
        true (array-like): Ground truth values.
        preds (array-like, optional): Predicted values. Defaults to None.
        name (str): Output file path for the visualization. Defaults to './pic/test.pdf'.
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    """
    Adjust predictions for anomaly detection tasks.
    
    This function refines binary anomaly predictions by ensuring
    that consecutive anomalies are properly marked.
    
    Args:
        gt (array-like): Ground truth binary labels (0: normal, 1: anomaly).
        pred (array-like): Predicted binary labels (0: normal, 1: anomaly).
        
    Returns:
        tuple: (adjusted_ground_truth, adjusted_predictions)
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            # Start of anomaly sequence
            anomaly_state = True
            # Mark preceding points as anomalies if they are actually anomalies
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Mark following points as anomalies if they are actually anomalies
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            # End of anomaly sequence
            anomaly_state = False
        if anomaly_state:
            # Continue marking as anomaly
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    Calculate classification accuracy.
    
    Args:
        y_pred (array-like): Predicted labels.
        y_true (array-like): True labels.
        
    Returns:
        float: Classification accuracy (proportion of correct predictions).
    """
    return np.mean(y_pred == y_true)
