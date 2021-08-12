
import glob
import argparse
from main import train
import os

import pandas as pd

from utils.utils import reduce_memory, save_processing_datas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_feature', type=str, default="../数据集/traindata/user_features_data/user_features_data.csv", help="user feature data path")
    parser.add_argument('--video_feature', type=str, default="../数据集/traindata/video_features_data/video_features_data.csv", help="video feature data path")
    parser.add_argument('--history_behavior', type=str, default="../数据集/traindata/history_behavior_data/*/*.csv", help="user history behavior data path")
    parser.add_argument('--test_file', type=str, default="../数据集/testdata/test.csv", help="test user file")
    parser.add_argument('--save_path', type=str, default="./datas", help="processing output directory")
    opt = parser.parse_args()

    # output directory.
    os.makedirs(opt.save_path, exist_ok=True)

    # load dataset and reduce memory by transform int32 to int8 and etc.
    user_features = reduce_memory(pd.read_csv(opt.user_feature, sep='\t')) # 5910800 x 8
    video_features = reduce_memory(pd.read_csv(opt.video_feature, sep='\t')) # 49731 x 10
    history_behavior = pd.concat([reduce_memory(pd.read_csv(x, sep='\t')) for x in glob.glob(opt.history_behavior)], axis=0)
    history_behavior = reduce_memory(history_behavior.sort_values(by=['pt_d', 'user_id'])) # 80276856 x 9

    test_data = reduce_memory(pd.read_csv(opt.test_file, sep=',')) # 2822180 x 2

    # Data processing.
    # Extract the historical behavior data of users in the test set.
    history_behavior = history_behavior[history_behavior['user_id'].isin(test_data['user_id'].unique())] # 20596355 x 9 
    # Extract the user's data on the 14th day.
    val_behavior = history_behavior[history_behavior['pt_d'] == 20210502]
    # Take the data from day 1 to day 13 as the training set.
    train_behavior = history_behavior[history_behavior['pt_d'] != 20210502]

    # Renamed for 14th day. 
    val_behavior = val_behavior.rename(columns={"watch_label": "val_watch", "is_share": "val_share"})

    # Merging the 14th data to the datas that is from day 1 to day 13. 
    train_behavior = pd.merge(train_behavior, 
                                val_behavior[['user_id', 'video_id', 'val_watch', 'val_share']], 
                                on=['user_id', 'video_id'], how='left')
    train_behavior['val_watch'] = train_behavior['val_watch'].fillna(0)
    train_behavior['val_share'] = train_behavior['val_share'].fillna(0)

    # Because there are 18 million pieces of data that have not been read (val_watch == 0), 
    # only 33000 pieces have been read (val_watch != 0). So, we sampled 50000 pieces from data that haven't been read.
    train_behavior = pd.concat([
        train_behavior[train_behavior['val_watch'] == 0].sample(50000),
        train_behavior[train_behavior['val_watch'] != 0]
    ])

    # Else analysis.
    train_user_behavior_agg = train_behavior.groupby('user_id').agg({
        'pt_d': ['count'],
        'video_id': ['nunique'],
        'is_watch': ['mean', 'max'],
        'is_share': ['mean', 'max'],
        'watch_label': ['nunique']
    })
    train_user_behavior_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in train_user_behavior_agg.columns.tolist()])
    train_user_behavior_agg = train_user_behavior_agg.reset_index()

    #history_behavior[(history_behavior['user_id'] == 2) & (history_behavior['video_id'] == 25469)]

    # save datas to specified folder.
    if save_processing_datas(train_behavior, os.path.join(opt.save_path, "train_dataset.csv")):
        print("\nsave train dataset success")
        
    if save_processing_datas(test_data, os.path.join(opt.save_path, "test_datatset.csv")):
        print("save test dataset success")
