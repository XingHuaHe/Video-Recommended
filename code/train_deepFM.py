
import os
import argparse

import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.models import DeepFM
from utils.dataset import DeepFMDataset

def train(opt: argparse.ArgumentParser) -> None:
    model_name, EPOCHS, batch_size, train_behavior_path, val_behavior_path, user_features_path = \
        opt.model, opt.epochs, opt.batch_size, opt.train_behavior, opt.val_behavior, opt.user_features,
    
    # device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # modeling
    if model_name == "DeepFM":
        model = DeepFM(cate_fea_nuniqs=[5910798+1, 50355+1, 34, 340, 1927], nume_fea_size=2)
    else:
        raise ValueError(f"Not such model {model_name}")
    model.to(device)

    # load dataset.
    train_behavior = pd.read_csv(train_behavior_path)
    val_behavior = pd.read_csv(val_behavior_path)
    user_features = pd.read_csv(user_features_path)

    # dataset.
    train_dataset = DeepFMDataset(train_behavior, user_features, train=True)
    test_dataset = DeepFMDataset(val_behavior, user_features, train=False)

    # dataloader.
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # loss function.
    wathch_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,2,2,2,2,2,2,2,2,2]).to(device))
    shaere_loss_fn = nn.BCEWithLogitsLoss()

    # optmizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # train.
    model.train()
    for epoch in range(EPOCHS):
        for i, data in tqdm.tqdm(enumerate(train_dataloader), ncols=100):
            # format data.
            # user_id = data['user_id']
            # video_id = data['video_id']
            # user_province = data['user_province']
            # user_city = data['user_city']
            # user_device = data['user_device']
            # user_age = data['user_age']
            # city_level = data['city_level']

            watch_label = data['watch_label']
            share_label = data['share_label']

            # clear grad.
            optimizer.zero_grad()

            # forward.
            wathch_pred, share_pred = model()

            # computed losses.
            loss = wathch_loss_fn(wathch_pred, watch_label) + shaere_loss_fn(share_pred, share_label)

            # backward.
            loss.backward()

            # update grad(parameter weight).
            optimizer.step()



if __name__ == "__main__":
    # test_data = pd.read_hdf('digix-data.hdf', 'test_data')
    # user_features = pd.read_hdf('digix-data.hdf', 'user_features')
    # video_features = pd.read_hdf('digix-data.hdf', 'video_features')
    # history_behavior = pd.read_hdf('digix-data.hdf', 'history_behavior')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_behavior', type=str, default="./datas/deepFm/train_behavior.csv", help="train_behavior path")
    parser.add_argument('--val_behavior', type=str, default="./datas/deepFm/val_behavior.csv", help="val_behavior path")
    parser.add_argument('--user_features', type=str, default="./datas/deepFm/user_features.csv", help="user_features path")
    parser.add_argument('--model', type=str, default="DeepFM", help="selected model class")
    parser.add_argument('--epochs', type=int, default=1, help="training epoch")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--outputs', type=str, default="./outputs/DeepFM", help="outputs directory")
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/DeepFM", help="checkpoint directory")
    parser.add_argument('--save', type=bool, default=True, help="whether to save checkpoint")
    opt = parser.parse_args()
    print(opt)

    # make directory.
    os.makedirs(opt.outputs, exist_ok=True)
    os.makedirs(opt.checkpoint, exist_ok=True)

    # trainging.
    train(opt)
