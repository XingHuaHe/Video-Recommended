
import argparse
import os
import pandas as pd

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.models import MLP
from utils.dataset import MLPDataset

def train(opt: argparse.ArgumentParser) -> None:
    model_name, EPOCHS, batch_size, train_dataset_path, test_dataset_path = \
        opt.model, opt.epochs, opt.batch_size, opt.train_dataset, opt.test_dataset

    # device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # modeling
    if model_name == "baseline":
        model = MLP()
    else:
        raise ValueError(f"Not such model {model_name}")
    model.to(device)

    # load dataset from directory.
    train_behavior = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    # dataset.
    train_dataset = MLPDataset(train_behavior, train=True)
    test_dataset = MLPDataset(test_data, train=False)

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
            feed_dict = {
                'user_id': data[0].long().to(device),
                'video_id': data[1].long().to(device),
            }
            watch_label = data[2].long().to(device)
            share_label = data[3].float().to(device)

            # clear grad.
            optimizer.zero_grad()

            # forward.
            wathch_pred, share_pred = model(feed_dict)

            # computed losses.
            loss = wathch_loss_fn(wathch_pred, watch_label) + shaere_loss_fn(share_pred, share_label)

            # backward.
            loss.backward()

            # update grad(parameter weight).
            optimizer.step()

            # 
            acc = ((wathch_pred.argmax(1) == watch_label)[watch_label != 0]).float().sum()
            acc /= (watch_label != 0).float().sum()
            if i % 50 == 0:
                print(f'epoch:{epoch}/{EPOCHS} \t {i}/{len(train_dataloader)} \t losses:{loss.item()} \
                     \t accuracy: {acc}', (wathch_pred.argmax(1) == watch_label).float().sum())
        
        if opt.save and epoch % 10 == 0 and epoch != 0:
            checkpoint = {
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint, f"{opt.checkpoint}/{model_name}_{epoch+1}.pth") # checkpoint_interval

    # predicted.
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
    # argument set.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default="./datas/train_dataset.csv", help="train data path")
    parser.add_argument('--test_dataset', type=str, default="./datas/test_datatset.csv", help="test data path")
    parser.add_argument('--model', type=str, default="baseline", help="selected model class")
    parser.add_argument('--epochs', type=int, default=1, help="training epoch")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--outputs', type=str, default="./outputs/baseline", help="outputs directory")
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/baseline", help="checkpoint directory")
    parser.add_argument('--save', type=bool, default=True, help="whether to save checkpoint")
    opt = parser.parse_args()
    print(opt)

    # make directory.
    os.makedirs(opt.outputs, exist_ok=True)
    os.makedirs(opt.checkpoint, exist_ok=True)

    # training.
    train(opt)