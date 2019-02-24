import os
from typing import List
import numpy as np
import h5py
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from experiments import get_files_list
from training import NetTrainer, BatchResult
import os
import h5py
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import List
from experiments import get_files_list
import numba
from line_profiler import LineProfiler
import time
import distance_mat
from mpl_toolkits.mplot3d import Axes3D


class HybMat40Ds(Dataset):
    def __init__(self, lbo_files: List[str], dist_files: List[str], pc_files: List[str], size: int):
        super().__init__()
        self.data_name = 'eigen_vec'
        self.size = size
        lbo_examples = []
        dist_examples = []
        self.mat_examples = []
        self.label_examples = []
        self.train = 'train' in lbo_files[0]
        for h5_file in lbo_files:
            current_data = self.load_h5(h5_file, self.data_name)
            current_data = current_data[:, :, 0:self.size]
            for i in range(current_data.shape[0]):
                lbo_examples.append(current_data[i, :, :])
        for h5_file in dist_files:
            current_data = self.load_h5(h5_file, self.data_name)
            current_data = current_data[:, :, :self.size]
            for i in range(current_data.shape[0]):
                dist_examples.append(current_data[i, :, :])
        assert (len(lbo_examples) == len(dist_examples))
        for lbo_eig, dist_eig in zip(lbo_examples, dist_examples):
            self.mat_examples.append(np.expand_dims(lbo_eig.transpose() @ dist_eig, axis=0))
        del lbo_examples
        del dist_examples

        for h5_file in pc_files:
            current_data = self.load_h5(h5_file, 'label')
            for i in range(current_data.shape[0]):
                self.label_examples.append(np.expand_dims(current_data[i, :], axis=0))
        assert (len(self.mat_examples) == len(self.label_examples))

    def __getitem__(self, index):
        item, label = self.get_numpy_data(index)
        #if self.train:
         #   item = self.rand_sign(item)
            # item = self.add_noise(item)
        #item_tensor = torch.from_numpy(item).float()
        #label_tensor = torch.from_numpy(label).long()
        return item, label

    def __len__(self):
        return len(self.mat_examples)

    def get_numpy_data(self, index):
        mat = self.mat_examples[index]
        label = self.label_examples[index]
        return mat, label

    def load_h5(self, h5_filename, data_name):
        f = h5py.File(h5_filename)
        data = f[data_name][:]
        return data

    def rand_sign(self, mat):
        sign1 = np.random.choice([-1, 1], size=self.size)
        sign2 = np.random.choice([-1, 1], size=self.size)
        sign1 = np.reshape(sign1, (self.size, 1))
        sign2 = np.reshape(sign2, (self.size, 1))
        mat = np.multiply(sign1.transpose(), mat)
        mat = np.multiply(mat, sign2)
        return mat

    def add_noise(self, mat):
        noise = np.random.normal(1, 0.0001, (self.size, self.size))
        return mat * noise


matrix_size = 32
bs_train, bs_test = 32, 32
train_files_pc = get_files_list(f'../data/modelnet40_ply_hdf5_2048/train_files.txt')
test_files_pc = get_files_list(f'../data/modelnet40_ply_hdf5_2048/test_files.txt')

train_files_lbo = get_files_list('../data/lbo_eig_2048/spectral_train_files.txt')
test_files_lbo = get_files_list('../data/lbo_eig_2048/spectral_test_files.txt')

train_files_dist = get_files_list('../data/dist_eig_2048/spectral_train_files.txt')
test_files_dist = get_files_list('../data/dist_eig_2048/spectral_test_files.txt')

# ds_train = Spectral40Ds(train_files, size=matrix_size)
# ds_test = Spectral40Ds(test_files, size=matrix_size)

ds_train = HybMat40Ds(lbo_files=train_files_lbo, dist_files=train_files_dist, pc_files=train_files_pc, size=matrix_size)

size = 32
for i in range(5):
    mat, label = ds_train.__getitem__(index=i)
    mat = mat[0, :, :]
    #print(f'mat.shape={mat.shape}\nmat=\n{mat}')
    plt.figure(i)
    plt.imshow(mat, extent=[0, 1, 0, 1])
    #plt.imshow(mat)
    #plt.imshow(np.abs(mat1)-np.abs(mat2), extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.title(f'label')
plt.show()