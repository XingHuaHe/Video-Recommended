from sys import path
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader


class VedioRemDataset(Dataset):
    def __init__(self, X: DataFrame or np.ndarray, Y: DataFrame or np.ndarray, train: bool = True) -> None:
        super().__init__()

        if X.__class__ is pd.DataFrame:
            self.datas = X.to_numpy()
            self.labels = Y.to_numpy()

        elif X.__class__ is np.ndarray:
            self.datas = X
            self.labels = Y

    def __getitem__(self, index) -> Tuple:
        x = self.X[index]
        return super().__getitem__(index)


class MLPDataset(Dataset):
    def __init__(self, history_behavior: DataFrame, train: bool = True) -> None:
        self.history_behavior = history_behavior
        self.train = train
        
    def __getitem__(self, index: int) -> Tuple:
        user_id = self.history_behavior.iloc[index]['user_id']
        video_id = self.history_behavior.iloc[index]['video_id']
                
        if self.train:
            watch_label = self.history_behavior.iloc[index]['val_watch']
            share_label = self.history_behavior.iloc[index]['val_share']

            watch_label = int(watch_label)
            share_label = int(share_label)
            
            return user_id, video_id, \
                torch.from_numpy(np.array(watch_label)), \
                torch.from_numpy(np.array([share_label]))
        else:
            return user_id, video_id
        
    def __len__(self):
        return len(self.history_behavior)

class DeepFMDataset(Dataset):
    def __init__(self, history_behavior: DataFrame, user_features: DataFrame, train: bool = True) -> None:
        self.train = train
        self.history_behavior = history_behavior
        self.user_features = user_features
        
    def __getitem__(self, index: int) -> Dict:
        user_id = self.history_behavior.iloc[index]['user_id']
        video_id = self.history_behavior.iloc[index]['video_id']
        
        feed_dict = {
            'user_id': user_id,
            'video_id': video_id,
            'user_province': self.user_features.loc[user_id]['province'],
            'user_city': self.user_features.loc[user_id]['city'],
            'user_device': self.user_features.loc[user_id]['device_name'],
            'user_age': self.user_features.loc[user_id]['age'],
            'city_level': self.user_features.loc[user_id]['city_level'],
        }
        
        if self.train:
            watch_label = self.history_behavior.iloc[index]['watch_label']
            share_label = self.history_behavior.iloc[index]['is_share']

            feed_dict['watch_label'] = watch_label
            feed_dict['share_label'] = share_label
            return feed_dict
        else:
            return feed_dict
        
    def __len__(self):
        return len(self.history_behavior)