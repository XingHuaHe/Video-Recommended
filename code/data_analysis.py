# %%
import glob

import pandas as pd

from utils.utils import reduce_memory

# %%

# load dataset and reduce memory by transform int32 to int8 and etc.
# user_features = reduce_memory(pd.read_csv("../数据集/traindata/user_features_data/user_features_data.csv", sep='\t')) # 5910800 x 8
# video_features = reduce_memory(pd.read_csv("../数据集/traindata/video_features_data/video_features_data.csv", sep='\t')) # 49731 x 10
history_behavior = pd.concat([reduce_memory(pd.read_csv(x, sep='\t')) for x in glob.glob("../数据集/traindata/history_behavior_data/*/*.csv")], axis=0)
history_behavior = reduce_memory(history_behavior.sort_values(by=['pt_d', 'user_id'])) # 80276856 x 9

# test_data = reduce_memory(pd.read_csv("../数据集/testdata/test.csv", sep=',')) # 2822180 x 2

# %%
# 获得被用户分析的历史数据
share_history_behavior = history_behavior[history_behavior['is_share'] == 1]
nshare_history_behavior = history_behavior[history_behavior['is_share'] == 0]
# 获得被分享的视频 id
video_share_id = share_history_behavior['video_id'].unique()
video_nshare_id = nshare_history_behavior['video_id'].unique()

# 获得被分享的视频的详细信息
video_features = reduce_memory(pd.read_csv("../数据集/traindata/video_features_data/video_features_data.csv", sep='\t'))
share_video_features = video_features[video_features['video_id'].isin(video_share_id)]
nshare_video_features = video_features[video_features['video_id'].isin(video_nshare_id)]

# 被分享的视频的第二标签的特征集合
share_video_second_class = share_video_features['video_second_class'].to_numpy()
video_share_labels_set = set()
for item in share_video_second_class:
    try:
        for x in item.split(','):
            video_share_labels_set.add(x)
    except:
        continue

# %%
# 不被分享的视频的第二标签的特征集合
share_video_second_class = nshare_video_features['video_second_class'].to_numpy()
video_nshare_labels_set = set()
for item in share_video_second_class:
    try:
        for x in item.split(','):
            video_nshare_labels_set.add(x)
    except:
        continue
# %%
# 交集
intersection = video_nshare_labels_set.intersection(video_share_labels_set)
difference = video_nshare_labels_set.difference(video_share_labels_set)

"""结论：
（1）被分享的视频，它的标签，均包含在未被分享的视频当中，即分享有的，为分享都有
（2）未被分享的大部分含除分享外的标签

因此：可以将含分享标签但不含未分享标签的，划分为类别 1
    将含未分享标签的，划分为类别 0
"""
# %%
test_data = reduce_memory(pd.read_csv("../数据集/testdata/test.csv", sep=',')) # 2822180 x 2
history_behavior = history_behavior[history_behavior['user_id'].isin(test_data['user_id'].unique())]

# %%
new_history_behavior = pd.merge(history_behavior, video_features[video_features['video_id'].isin(history_behavior['video_id'])][['video_id' ,'video_second_class']], on=['video_id'], how='left')

# %%


val_behavior = history_behavior[history_behavior['pt_d'] == 20210502] # 1-14天数据，取第14天的数据作为验证集
train_behavior = history_behavior[history_behavior['pt_d'] != 20210502] # 取1-13天的数据作为训练集