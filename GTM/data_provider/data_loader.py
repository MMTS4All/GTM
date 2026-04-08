"""
Time series dataset loaders for various benchmarks and custom datasets.

This module provides dataset classes for loading and preprocessing time series data
from popular benchmarks like ETT, weather, electricity, traffic, and anomaly detection
datasets like PSM, MSL, SMAP, SMD, and SWaT.
"""

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    """
    ETT hourly dataset loader.
    
    This class loads and preprocesses the ETT (Electricity Transformer Temperature)
    hourly dataset for time series forecasting tasks.
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        """
        Initialize the ETT hourly dataset.
        
        Args:
            root_path (str): Root directory containing the dataset files.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
            size (list, optional): [sequence_length, prediction_length]. Defaults to None.
            features (str): Feature selection mode ('S', 'M', 'MS'). Defaults to 'S'.
            data_path (str): Path to the dataset CSV file. Defaults to 'ETTh1.csv'.
            target (str): Target column name. Defaults to 'OT'.
            scale (bool): Whether to apply standard scaling. Defaults to True.
            timeenc (int): Time encoding method (0 or 1). Defaults to 0.
            freq (str): Time frequency. Defaults to 'h'.
        """
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess the dataset."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (seq_x, seq_y, time_gra) where:
                - seq_x: Input sequence
                - seq_y: Target sequence
                - time_gra: Time granularity information
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        #时间粒度信息[ms,s,min,hour,day]
        gra = [0,0,0,1,0]
        return seq_x, seq_y,gra

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Inverse transform scaled data to original scale.
        
        Args:
            data (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    """
    ETT minute-level dataset loader.
    
    This class loads and preprocesses the ETT (Electricity Transformer Temperature)
    minute-level dataset for high-resolution time series forecasting tasks.
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        """
        Initialize the ETT minute-level dataset.
        
        Args:
            root_path (str): Root directory containing the dataset files.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
            size (list, optional): [sequence_length, prediction_length]. Defaults to None.
            features (str): Feature selection mode ('S', 'M', 'MS'). Defaults to 'S'.
            data_path (str): Path to the dataset CSV file. Defaults to 'ETTm1.csv'.
            target (str): Target column name. Defaults to 'OT'.
            scale (bool): Whether to apply standard scaling. Defaults to True.
            timeenc (int): Time encoding method (0 or 1). Defaults to 0.
            freq (str): Time frequency. Defaults to 'h'.
        """
        # size [seq_len, label_len, pred_len]
        # self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess the dataset."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (seq_x, seq_y, time_gra) where:
                - seq_x: Input sequence
                - seq_y: Target sequence
                - time_gra: Time granularity information
        """
        s_begin = index*1
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        #时间粒度信息[ms,s,min,hour,day]
        gra = [0,0,15,0,0]
        return seq_x, seq_y,gra

    def __len__(self):
        """Get the number of samples in the dataset."""
        return (len(self.data_x) - self.seq_len - self.pred_len)//1 + 1

    def inverse_transform(self, data):
        """
        Inverse transform scaled data to original scale.
        
        Args:
            data (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)


class Dataset_weather(Dataset):
    """
    Weather dataset loader.
    
    This class loads and preprocesses weather time series data for forecasting tasks.
    """
    
    def __init__(self,
                 root_path="/home/liangxijie1/DMLoRA/dataset/weather",
                 flag="train",
                 size=None,
                 features="M",
                 data_path='weather.csv',
                 target='OT',
                 scale=True,
                 inverse=False,
                 timeenc=0,
                 freq='t'):
        """
        Initialize the weather dataset.
        
        Args:
            root_path (str): Root directory containing the dataset files.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
            size (list, optional): [sequence_length, prediction_length]. Defaults to None.
            features (str): Feature selection mode ('S', 'M', 'MS'). Defaults to 'M'.
            data_path (str): Path to the dataset CSV file. Defaults to 'weather.csv'.
            target (str): Target column name. Defaults to 'OT'.
            scale (bool): Whether to apply standard scaling. Defaults to True.
            inverse (bool): Whether to inverse transform data. Defaults to False.
            timeenc (int): Time encoding method (0 or 1). Defaults to 0.
            freq (str): Time frequency. Defaults to 't'.
        """
        super(Dataset_weather, self).__init__()
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess the dataset."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        train_length = int(len(df_raw) * .7)
        test_length = int(len(df_raw) * .2)
        val_length = len(df_raw) - test_length - train_length

        border1s = [0, train_length - self.seq_len, len(df_raw) - test_length - self.seq_len]
        border2s = [train_length, train_length + val_length, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[0:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, item):
        """
        Get a single data sample.
        
        Args:
            item (int): Sample index.
            
        Returns:
            tuple: (seq_x, seq_y, time_gra) where:
                - seq_x: Input sequence
                - seq_y: Target sequence
                - time_gra: Time granularity information
        """
        s_begin = item
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        gra = [0, 0, 10, 0, 0]
        return seq_x, seq_y,gra

    def __len__(self):
        """Get the number of samples in the dataset."""
        return (len(self.data_x) - self.seq_len - self.pred_len) + 1

    def inverse_transform(self, data):
        """
        Inverse transform scaled data to original scale.
        
        Args:
            data (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)


class Dataset_elc(Dataset):
    """
    Electricity consumption dataset loader.
    
    This class loads and preprocesses electricity consumption time series data
    for forecasting tasks.
    """
    
    def __init__(self,
                 root_path="/home/liangxijie1/DMLoRA/dataset/electricity",
                 flag="train",
                 size=None,
                 features="M",
                 data_path='electricity.csv',
                 target='OT',
                 scale=True,
                 inverse=False,
                 timeenc=0,
                 freq='t'):
        """
        Initialize the electricity dataset.
        
        Args:
            root_path (str): Root directory containing the dataset files.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
            size (list, optional): [sequence_length, prediction_length]. Defaults to None.
            features (str): Feature selection mode ('S', 'M', 'MS'). Defaults to 'M'.
            data_path (str): Path to the dataset CSV file. Defaults to 'electricity.csv'.
            target (str): Target column name. Defaults to 'OT'.
            scale (bool): Whether to apply standard scaling. Defaults to True.
            inverse (bool): Whether to inverse transform data. Defaults to False.
            timeenc (int): Time encoding method (0 or 1). Defaults to 0.
            freq (str): Time frequency. Defaults to 't'.
        """
        super(Dataset_elc, self).__init__()
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess the dataset."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        train_length = int(len(df_raw) * .7)
        test_length = int(len(df_raw) * .2)
        val_length = len(df_raw) - test_length - train_length
        border1s = [0, train_length - self.seq_len, len(df_raw) - test_length - self.seq_len]
        border2s = [train_length, train_length + val_length, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, item):
        """
        Get a single data sample.
        
        Args:
            item (int): Sample index.
            
        Returns:
            tuple: (seq_x, seq_y, time_gra) where:
                - seq_x: Input sequence
                - seq_y: Target sequence
                - time_gra: Time granularity information
        """
        s_begin = item
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        time_gra = [0,0,0,1,0]
        return seq_x, seq_y,time_gra

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Inverse transform scaled data to original scale.
        
        Args:
            data (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)


class Dataset_traffic(Dataset):
    """
    Traffic flow dataset loader.
    
    This class loads and preprocesses traffic flow time series data for forecasting tasks.
    """
    
    def __init__(
            self,
            root_path="/data/liangxijie/DMLoRA/dataset/traffic",
            flag="train",
            size=None,
            features="M",
            data_path="traffic.csv",
            target="OT",
            scale=True,
            inverse=False,
            timeenc=0,
            freq="t",
            cols=None,
    ):
        """
        Initialize the traffic dataset.
        
        Args:
            root_path (str): Root directory containing the dataset files.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
            size (list, optional): [sequence_length, prediction_length]. Defaults to None.
            features (str): Feature selection mode ('S', 'M', 'MS'). Defaults to 'M'.
            data_path (str): Path to the dataset CSV file. Defaults to 'traffic.csv'.
            target (str): Target column name. Defaults to 'OT'.
            scale (bool): Whether to apply standard scaling. Defaults to True.
            inverse (bool): Whether to inverse transform data. Defaults to False.
            timeenc (int): Time encoding method (0 or 1). Defaults to 0.
            freq (str): Time frequency. Defaults to 't'.
            cols (list, optional): Columns to select. Defaults to None.
        """
        super(Dataset_traffic, self).__init__()
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess the dataset."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        train_length = int(len(df_raw) * .7)
        test_length = int(len(df_raw) * .2)
        val_length = len(df_raw) - test_length - train_length
        border1s = [0, train_length - self.seq_len, len(df_raw) - test_length - self.seq_len]
        border2s = [train_length, train_length + val_length, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]: border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (x, y, time_gra) where:
                - x: Input sequence
                - y: Target sequence
                - time_gra: Time granularity information
        """
        i_begin = index
        i_sep = i_begin + self.seq_len
        i_end = i_sep + self.pred_len

        seq_x = self.data_x[i_begin:i_sep]
        seq_y = self.data_y[i_sep: i_end]

        x = torch.tensor(seq_x, dtype=torch.float)
        y = torch.tensor(seq_y, dtype=torch.float)
        # x_mark = torch.tensor(self.data_stamp[i_begin:i_sep], dtype=torch.int)

        # y[: self.seq_len - 1] = float("nan")
        gra = [0,0,0,1,0]
        return x, y,gra

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Inverse transform scaled data to original scale.
        
        Args:
            data (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    """
    Custom dataset loader for user-defined time series data.
    
    This class provides a flexible interface for loading custom time series datasets
    with configurable parameters for various forecasting scenarios.
    """
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        """
        Initialize the custom dataset.
        
        Args:
            args (argparse.Namespace): Configuration arguments.
            root_path (str): Root directory containing the dataset files.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
            size (list, optional): [sequence_length, label_length, prediction_length]. Defaults to None.
            features (str): Feature selection mode ('S', 'M', 'MS'). Defaults to 'S'.
            data_path (str): Path to the dataset CSV file. Defaults to 'ETTh1.csv'.
            target (str): Target column name. Defaults to 'OT'.
            scale (bool): Whether to apply standard scaling. Defaults to True.
            timeenc (int): Time encoding method (0 or 1). Defaults to 0.
            freq (str): Time frequency. Defaults to 'h'.
            seasonal_patterns (str, optional): Seasonal patterns. Defaults to None.
        """
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess the dataset."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (seq_x, seq_y, seq_x_mark, seq_y_mark) where:
                - seq_x: Input sequence
                - seq_y: Target sequence
                - seq_x_mark: Input time marks
                - seq_y_mark: Target time marks
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        Inverse transform scaled data to original scale.
        
        Args:
            data (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    """
    M4 competition dataset loader.
    
    This class loads and preprocesses the M4 competition time series data
    for benchmarking forecasting models.
    """
    
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        """
        Initialize the M4 dataset.
        
        Args:
            args (argparse.Namespace): Configuration arguments.
            root_path (str): Root directory containing the dataset files.
            flag (str): Data split ('train', 'val', 'test', 'pred'). Defaults to 'pred'.
            size (list, optional): [sequence_length, label_length, prediction_length]. Defaults to None.
            features (str): Feature selection mode ('S', 'M', 'MS'). Defaults to 'S'.
            data_path (str): Path to the dataset file. Defaults to 'ETTh1.csv'.
            target (str): Target column name. Defaults to 'OT'.
            scale (bool): Whether to apply standard scaling. Defaults to False.
            inverse (bool): Whether to inverse transform data. Defaults to False.
            timeenc (int): Time encoding method (0 or 1). Defaults to 0.
            freq (str): Time frequency. Defaults to '15min'.
            seasonal_patterns (str): Seasonal patterns. Defaults to 'Yearly'.
        """
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        """Read and preprocess the dataset."""
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
             dataset.values[dataset.groups == self.seasonal_patterns])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (insample, outsample, insample_mask, outsample_mask) where:
                - insample: Input sequence
                - outsample: Target sequence
                - insample_mask: Input mask
                - outsample_mask: Target mask
        """
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.timeseries)

    def inverse_transform(self, data):
        """
        Inverse transform scaled data to original scale.
        
        Args:
            data (np.ndarray): Scaled data.
            
        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    """
    PSM (Power Supply Monitoring) dataset loader for anomaly detection.
    
    This class loads and preprocesses the PSM dataset for anomaly detection tasks.
    """
    
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        """
        Initialize the PSM dataset.
        
        Args:
            args (argparse.Namespace): Configuration arguments.
            root_path (str): Root directory containing the dataset files.
            win_size (int): Window size for segmentation.
            step (int): Step size for sliding window. Defaults to 1.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (data, labels, time_gra) where:
                - data: Input data segment
                - labels: Anomaly labels
                - time_gra: Time granularity information
        """
        index = index * self.step
        time_gra = [0,0,1,0,0]
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size]),time_gra
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size]),time_gra
        elif (self.flag == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]),time_gra


class MSLSegLoader(Dataset):
    """
    MSL (Multi-source Selected Logs) dataset loader for anomaly detection.
    
    This class loads and preprocesses the MSL dataset for anomaly detection tasks.
    """
    
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        """
        Initialize the MSL dataset.
        
        Args:
            args (argparse.Namespace): Configuration arguments.
            root_path (str): Root directory containing the dataset files.
            win_size (int): Window size for segmentation.
            step (int): Step size for sliding window. Defaults to 1.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        # elif (self.flag == 'test'):
        #     return (self.test.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (data, labels, time_gra) where:
                - data: Input data segment
                - labels: Anomaly labels
                - time_gra: Time granularity information
        """
        index = index * self.step
        time_gra = [0,0,1,0,0]
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size]),time_gra
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size]),time_gra
        # elif (self.flag == 'test'):
        #     return np.float32(self.test[index:index + self.win_size]), np.float32(
        #         self.test_labels[index:index + self.win_size],time_gra)
        elif (self.flag == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]),time_gra


class SMAPSegLoader(Dataset):
    """
    SMAP (Soil Moisture Active Passive) dataset loader for anomaly detection.
    
    This class loads and preprocesses the SMAP satellite telemetry data
    for anomaly detection tasks.
    """
    
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        """
        Initialize the SMAP dataset.
        
        Args:
            args (argparse.Namespace): Configuration arguments.
            root_path (str): Root directory containing the dataset files.
            win_size (int): Window size for segmentation.
            step (int): Step size for sliding window. Defaults to 1.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (data, labels, time_gra) where:
                - data: Input data segment
                - labels: Anomaly labels
                - time_gra: Time granularity information
        """
        index = index * self.step
        time_gra = [0, 0, 1, 0, 0]
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]), time_gra
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]), time_gra
        elif (self.flag == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), time_gra


class SMDSegLoader(Dataset):
    """
    SMD (Server Machine Dataset) loader for anomaly detection.
    
    This class loads and preprocesses the SMD dataset for anomaly detection tasks.
    """
    
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        """
        Initialize the SMD dataset.
        
        Args:
            args (argparse.Namespace): Configuration arguments.
            root_path (str): Root directory containing the dataset files.
            win_size (int): Window size for segmentation.
            step (int): Step size for sliding window. Defaults to 100.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        """
        Number of images in the object dataset.
        
        Returns:
            int: Number of samples.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (data, labels, time_gra) where:
                - data: Input data segment
                - labels: Anomaly labels
                - time_gra: Time granularity information
        """
        index = index * self.step
        time_gra = [0, 0, 1, 0, 0]
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]), time_gra
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]), time_gra
        elif (self.flag == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), time_gra

class SWATSegLoader(Dataset):
    """
    SWaT (Secure Water Treatment) dataset loader for anomaly detection.
    
    This class loads and preprocesses the SWaT dataset for industrial control system
    anomaly detection tasks.
    """
    
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        """
        Initialize the SWaT dataset.
        
        Args:
            args (argparse.Namespace): Configuration arguments.
            root_path (str): Root directory containing the dataset files.
            win_size (int): Window size for segmentation.
            step (int): Step size for sliding window. Defaults to 1.
            flag (str): Data split ('train', 'val', 'test'). Defaults to 'train'.
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = np.load(root_path + "/swat_train.npy")
        test_data = np.load(root_path + "/swat_test.npy")

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val =test_data
        self.test_labels = np.load(root_path + "/swat_labels.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        
        Returns:
            int: Number of samples.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index.
            
        Returns:
            tuple: (data, labels, time_gra) where:
                - data: Input data segment
                - labels: Anomaly labels
                - time_gra: Time granularity information
        """
        index = index * self.step
        time_gra = [0, 1, 0, 0, 0]
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]), time_gra
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[0:self.win_size]), time_gra
        # elif (self.flag == 'test'):
        #     return np.float32(self.test[index:index + self.win_size]), np.float32(
        #         self.test_labels[index:index + self.win_size],time_gra)
        elif (self.flag == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[
                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), time_gra



