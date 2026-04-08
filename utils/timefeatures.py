"""
Time feature extraction utilities for time series modeling.

This module provides utilities for extracting temporal features from datetime indices,
which are commonly used as input features for time series forecasting models.
The implementation is adapted from GluonTS with normalized encoding between [-0.5, 0.5].
"""

# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    Base class for time feature extraction.
    
    This abstract base class defines the interface for extracting temporal features
    from pandas DatetimeIndex objects.
    """
    
    def __init__(self):
        """Initialize the time feature extractor."""
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract time features from datetime index.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Extracted time features.
        """
        pass

    def __repr__(self):
        """String representation of the time feature class."""
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract second of minute feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized second of minute values in range [-0.5, 0.5].
        """
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract minute of hour feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized minute of hour values in range [-0.5, 0.5].
        """
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract hour of day feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized hour of day values in range [-0.5, 0.5].
        """
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract day of week feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized day of week values in range [-0.5, 0.5].
        """
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract day of month feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized day of month values in range [-0.5, 0.5].
        """
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract day of year feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized day of year values in range [-0.5, 0.5].
        """
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract month of year feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized month of year values in range [-0.5, 0.5].
        """
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract week of year feature.
        
        Args:
            index (pd.DatetimeIndex): Datetime index to extract features from.
            
        Returns:
            np.ndarray: Normalized week of year values in range [-0.5, 0.5].
        """
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    
    This function maps frequency strings to appropriate time feature extractors based on
    the granularity of the data.
    
    Args:
        freq_str (str): Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
        
    Returns:
        List[TimeFeature]: List of time feature extractor classes appropriate for the frequency.
        
    Raises:
        RuntimeError: If the frequency string is not supported.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    """
    Extract time features from datetime index for the given frequency.
    
    This function applies appropriate time feature extractors to a datetime index
    and returns a vertically stacked array of features.
    
    Args:
        dates (pd.DatetimeIndex): Datetime index to extract features from.
        freq (str): Frequency string. Defaults to 'h' (hourly).
        
    Returns:
        np.ndarray: Vertically stacked array of extracted time features.
    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
