import numpy as np
import csv
import pandas as pd
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
import random


def read_make_set(dataset_path):
    
    data_set = list()
    with open(dataset_path, newline='') as csvfile:
        line_reader = csv.reader(csvfile, delimiter=',')
        one_set = list()
        for element in line_reader:
            line_list = list()
            for ele in element:
                if ele == "": # 빈칸일 경우
                    ele = 0
                line_list.append(ele)
            one_set.append(line_list)
        data_set.append(one_set)
    data_set = np.concatenate(data_set, axis=0)

    return data_set


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):

        data_set = read_make_set(dataset_path) 
        random.seed(1234)
        random.shuffle(data_set[1:,:])

        data = data_set[1:, 3:].astype(np.float32)
        target = data_set[1:, :3].astype(np.float32)
        
        # normalize
        # data_normed = data / data.max(axis=0)
        # data = normalize(data, axis=0, norm='max')
        # https://mkjjo.github.io/python/2019/01/10/scaler.html
        robustScaler = RobustScaler()
        print(robustScaler.fit(data))
        data = robustScaler.transform(data)
        
        
        self.data = torch.FloatTensor(data)
        self.target = torch.FloatTensor(target)

    def __getitem__(self, index):

        return self.data[index], self.target[index]

    def __len__(self):

        return self.data.size(0)

def myCollate(batch):

    data = torch.stack([item[0] for item in batch], 0) # [batch, 140]
    target = torch.stack([item[1] for item in batch], 0) # [batch, 3]

    return [data, target]

def getDataLoader(train_set, validation_set, test_set):
    
    train_loader = DataLoader(train_set, batch_size=96, num_workers=0, collate_fn=myCollate)
    validation_loader = DataLoader(validation_set, batch_size=1, num_workers=0, collate_fn=myCollate)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, collate_fn=myCollate)

    return train_loader, validation_loader, test_loader