"""
    数据处理，并保存为模型训练数据格式
"""

# %
import glob, gc
import argparse
import os
import pandas as pd
from pandas.core.frame import DataFrame

import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import reduce_memory
from utils.models import MLP
from utils.dataset import MLPDataset

def train(opt: argparse.ArgumentParser, epochs: int, train_behavior: DataFrame, test_data: DataFrame, ):
    # device.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # model.
    model = MLP()
    model.to(device)

    # dataset and dataloader.
    train_dataset = MLPDataset(train_behavior)
    test_dataset = MLPDataset(test_data, train=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # loss function.
    wathch_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,2,2,2,2,2,2,2,2,2]).to(device))
    shaere_loss_fn = nn.BCEWithLogitsLoss()

    # optmizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # train.
    model.train()
    for epoch in range(epochs):
        for i, data in tqdm.tqdm(enumerate(train_dataloader), ncols=100):
            feed_dict = {
                'user_id': data[0].long().to(device),
                'video_id': data[1].long().to(device),
            }
            watch_label = data[2].long().to(device)
            share_label = data[3].float().to(device)

            optimizer.zero_grad()
            wathch_pred, share_pred = model(feed_dict)
            loss = wathch_loss_fn(wathch_pred, watch_label) + shaere_loss_fn(share_pred, share_label)

            loss.backward()
            optimizer.step()

            acc = ((wathch_pred.argmax(1) == watch_label)[watch_label != 0]).float().sum()
            acc /= (watch_label != 0).float().sum()

            if i % 10 == 0:
                print(f'{i}/{len(train_dataloader)} \t losses:{loss.item()} \t accuracy: {acc}', (wathch_pred.argmax(1) == watch_label).float().sum())
                # print(wathch_pred.argmax(1))
            
            break
        
        if opt.save and epoch % 5 == 0 and epoch != 0:
            checkpoint = {
                'model': model,
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint, f"{opt.outputs}/MLP_{epoch+1}.pth") # checkpoint_interval

        break

    # result predicted.
    test_watch = []
    test_share = []
    with torch.no_grad():
        for data in tqdm.tqdm(test_dataloader):
            feed_dict = {
                'user_id': data[0].long().to(device),
                'video_id': data[1].long().to(device),
            }
            wathch_pred, share_pred = model(feed_dict)
            
            test_watch += list(wathch_pred.argmax(1).cpu().data.numpy())
            test_share += list((share_pred.sigmoid() > 0.5).int().cpu().data.numpy().flatten())

    test_data['watch_label'] = test_watch
    test_data['is_share'] = test_share

    test_data.to_csv(f'{opt.outputs}/submission.csv', index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_feature', type=str, default="../数据集/traindata/user_features_data/user_features_data.csv", help="user feature data path")
    parser.add_argument('--video_feature', type=str, default="../数据集/traindata/video_features_data/video_features_data.csv", help="video feature data path")
    parser.add_argument('--history_behavior', type=str, default="../数据集/traindata/history_behavior_data/*/*.csv", help="user history behavior data path")
    parser.add_argument('--test_file', type=str, default="../数据集/testdata/test.csv", help="test user file")

    parser.add_argument('--outputs', type=str, default="./outputs", help="outputs directory")

    parser.add_argument('--epochs', type=int, default=10, help="training epoch")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--save', type=bool, default=True, help="whether to save checkpoint")

    opt = parser.parse_args()

    # make directory.
    os.makedirs(opt.outputs, exist_ok=True)

    # load datas.
    user_features = reduce_memory(pd.read_csv(opt.user_feature, sep='\t'))
    video_features = reduce_memory(pd.read_csv(opt.video_feature, sep='\t'))
    history_behavior = pd.concat([reduce_memory(pd.read_csv(x, sep='\t')) for x in glob.glob(opt.history_behavior)], axis=0)
    history_behavior = reduce_memory(history_behavior.sort_values(by=['pt_d', 'user_id']))

    test_data = reduce_memory(pd.read_csv(opt.test_file, sep=','))

    # %
    history_behavior = history_behavior[history_behavior['user_id'].isin(test_data['user_id'].unique())]
    val_behavior = history_behavior[history_behavior['pt_d'] == 20210502] # 1-14天数据，取第14天的数据作为验证集
    train_behavior = history_behavior[history_behavior['pt_d'] != 20210502] # 取1-13天的数据作为训练集

    val_behavior = val_behavior.rename(columns={"watch_label": "val_watch", "is_share": "val_share"})

    # %
    train_behavior = pd.merge(train_behavior, 
                              val_behavior[['user_id', 'video_id', 'val_watch', 'val_share']], 
                              on=['user_id', 'video_id'], how='left')

    # %
    train_behavior['val_watch'] = train_behavior['val_watch'].fillna(0)
    train_behavior['val_share'] = train_behavior['val_share'].fillna(0)
    # 
    train_behavior = pd.concat([
        train_behavior[train_behavior['val_watch'] == 0].sample(50000),
        train_behavior[train_behavior['val_watch'] != 0]
    ])

    # train_behavior['val_watch'].value_counts()

    # %
    train_user_behavior_agg = train_behavior.groupby('user_id').agg({
        'pt_d': ['count'],
        'video_id': ['nunique'],
        'is_watch': ['mean', 'max'],
        'is_share': ['mean', 'max'],
        'watch_label': ['nunique']
    })

    train_user_behavior_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in train_user_behavior_agg.columns.tolist()])
    train_user_behavior_agg = train_user_behavior_agg.reset_index()

    history_behavior[(history_behavior['user_id'] == 2) & (history_behavior['video_id'] == 25469)]

    train(opt, opt.epochs, train_behavior, test_data)
