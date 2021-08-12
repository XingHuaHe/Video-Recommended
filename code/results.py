"""加载模型，获得输出结果"""

import argparse

import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.utils import reduce_memory
from utils.dataset import MLPDataset
from utils.models import MLP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default="../数据集/testdata/test.csv", help="test user file")
    parser.add_argument('--checkpoint', type=str, default="./outputs/MLP_20.pth", help="checkpoint files")
    parser.add_argument('--outputs', type=str, default="./outputs", help="outputs directory")
    opt = parser.parse_args()

    # device.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # load data.
    test_data = reduce_memory(pd.read_csv(opt.test_file, sep=','))

    # dataset and dataloader.
    test_dataset = MLPDataset(test_data, train=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # model.
    model = MLP()
    model.to(device)

    # load state dict.
    state_dict = torch.load(opt.checkpoint)['state_dict']
    model.load_state_dict(state_dict)

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