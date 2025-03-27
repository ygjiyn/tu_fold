import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


class Rna8fSubsetsDataset(Dataset):
    '''
    - rna8f_root
        - 150
            - train
                - sample_1_x.npy
                - sample_1_y.npy
                - ...
            - valid
            - test
        - 500
            - ...
    '''
    def __init__(self, dataset_root_path, sub_dataset_list, dataset_type, max_seq_len):
        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        assert dataset_type in ('train', 'valid', 'test')

        self.dataset_type = dataset_type

        for sub_dataset in sub_dataset_list:
            sample_root_path = os.path.join(dataset_root_path, sub_dataset, dataset_type)
            for file_name in os.listdir(sample_root_path):
                if file_name.endswith('_x.npy'):
                    y_file_name = file_name[:-6] + '_y.npy'
                    self.all_x_list.append(np.load(os.path.join(sample_root_path, file_name)))
                    self.all_y_list.append(np.load(os.path.join(sample_root_path, y_file_name)))

    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]

        # padding token: 0
        x = F.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=0)

        if self.dataset_type == 'train':
            # class indices target
            # note that while evaluating, we want a binary matrix label
            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))
            if y_npy.size > 0:
                y[y_npy[:, 0], y_npy[:, 1]] = 1.0

        return x, y


def get_one_sample_for_predict(x_npy, max_seq_len):
    x = F.pad(torch.from_numpy(x_npy), pad=(0, max_seq_len - x_npy.size), value=0)
    return x.unsqueeze(0)


class Rna8fByRnaFamilyDataset(Dataset):

    def __init__(self, dataset_root_path, sub_dataset_list, dataset_type, max_seq_len, 
                 rna_family_name_list):
        assert dataset_type in ('train', 'valid', 'test')
        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        self.dataset_type = dataset_type

        for sub_dataset in sub_dataset_list:
            sub_sub_dataset_root_path = os.path.join(dataset_root_path, sub_dataset)

            for sub_sub_dataset in rna_family_name_list:
                sample_root_path = os.path.join(sub_sub_dataset_root_path, sub_sub_dataset)
                if not os.path.exists(sample_root_path):
                    continue
                
                main_name_set = set()

                for file_name in os.listdir(sample_root_path):
                    if file_name.endswith('.npy'):
                        main_name_set.add(file_name[:-6])
            
                for main_name in main_name_set:
                    x_name = main_name + '_x.npy'
                    y_name = main_name + '_y.npy'
                    self.all_x_list.append(np.load(os.path.join(sample_root_path, x_name)))
                    self.all_y_list.append(np.load(os.path.join(sample_root_path, y_name)))

    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]

        # padding token: 0
        x = F.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=0)

        if self.dataset_type == 'train':
            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))
            if y_npy.size > 0:
                y[y_npy[:, 0], y_npy[:, 1]] = 1.0

        return x, y


class SynthesizedDataset(Dataset):

    def __init__(self, dataset_root_path, dataset_type, max_seq_len):
        assert dataset_type in ('train', 'valid', 'test')
        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        self.dataset_type = dataset_type

        main_name_set = set()

        for file_name in os.listdir(dataset_root_path):
            if file_name.endswith('.npy'):
                main_name_set.add(file_name[:-6])
            
        for main_name in main_name_set:
            x_name = main_name + '_x.npy'
            y_name = main_name + '_y.npy'
            self.all_x_list.append(np.load(os.path.join(dataset_root_path, x_name)))
            self.all_y_list.append(np.load(os.path.join(dataset_root_path, y_name)))

    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]

        # padding token: 0
        x = F.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=0)

        if self.dataset_type == 'train':
            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))
            if y_npy.size > 0:
                y[y_npy[:, 0], y_npy[:, 1]] = 1.0

        return x, y


class MergedDataset(Dataset):
    def __init__(self, dataset_list, dataset_type, max_seq_len):
        assert dataset_type in ('train', 'valid', 'test')
        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        self.dataset_type = dataset_type

        for dataset in dataset_list:
            self.all_x_list.extend(dataset.all_x_list)
            self.all_y_list.extend(dataset.all_y_list)
    
    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]

        # padding token: 0
        x = F.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=0)

        if self.dataset_type == 'train':
            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))
            if y_npy.size > 0:
                y[y_npy[:, 0], y_npy[:, 1]] = 1.0

        return x, y


def get_one_sample_for_eval(x_npy, y_npy, max_seq_len):
    # padding token: 0
    x = F.pad(torch.from_numpy(x_npy), pad=(0, max_seq_len - x_npy.size), value=0)
    y = torch.zeros((max_seq_len, max_seq_len))
    if y_npy.size > 0:
        y[y_npy[:, 0], y_npy[:, 1]] = 1.0
    return x.unsqueeze(0), y.unsqueeze(0)




if __name__ == '__main__':
    pass
