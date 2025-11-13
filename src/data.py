import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm

# model parameters
MAP_CUTOFF = 14
DIST_NORM = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


class ProDataset(Dataset):
    def __init__(
            self,
            pe_dim,
            dataset_name="train",
    ):
        self.transform = T.AddRandomWalkPE(walk_length=pe_dim, attr_name="pe")

        file_map = {
            "train": "train.pkl",
            'test_ALL': 'test.pkl',
            'test_PM': 'test_CB_PM.pkl',
            'test_P2RANK': 'test_CB_P2RANK.pkl',
        }
        file_name = file_map[dataset_name]
        file_path = '../dataset/graph'

        print("Loading Graph List ...")
        with open(os.path.join(file_path,file_name), "rb") as f:
            Graph_list = pickle.load(f)

        print("Transforming Graph List ...")
        self.Graph_list = []
        for graph in tqdm(Graph_list):
            assert graph.num_nodes == len(graph.y)
            self.Graph_list.append(self.transform(graph))

    def __getitem__(self, index):
        return self.Graph_list[index]

    def __len__(self):
        return len(self.Graph_list)


def get_train_validation_data_loaders(train_dataset, batch_size, train_idx, valid_idx, num_workers):
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, drop_last=False)

    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, drop_last=False)

    return train_loader, valid_loader

if __name__ == '__main__':
    train_dataset = ProDataset(30,'train')
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print(dataloader)
    for i in dataloader:
        print(i.x.shape,i.xyz_feats.shape)