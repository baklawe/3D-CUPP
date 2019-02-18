import os
import h5py
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from laplacian import ModelNet40Base
from typing import List
from experiments import get_files_list
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from training import NetTrainer


class CreateSpectralDist(ModelNet40Base):
    """
    This class helps to create the eigenfunction & eigenvalues of the LBO
    """
    def __init__(self, h5_files: List[str], num_nbrs: int, num_eigen: int):
        super().__init__(h5_files=h5_files)
        self.num_nbrs = num_nbrs
        self.num_eigen = num_eigen
        self.num_points = 1024

    def __getitem__(self, index):
        """
        :return: eigenfunctions & eigenvalues associated with the point cloud.
        """
        pc, label = self.get_pc(index)
        src, target, edges = self.get_knn(pc)
        L, D = self.create_laplacian(src, target)
        dist_mat = self.create_distance_mat(edges)
        spectral_dist = self.get_spectral_dist(L, D, dist_mat)
        return spectral_dist, label

    def __len__(self):
        return self.tot_examples

    def get_knn(self, pc):
        # +1 including the point itself
        nbrs = NearestNeighbors(n_neighbors=self.num_nbrs + 1, algorithm='auto', metric='euclidean').fit(pc)
        distances, indices = nbrs.kneighbors(pc)  # (N, num_nbrs + 1)
        src = np.tile(indices[:, 0], reps=self.num_nbrs)  # (N * num_nbrs,)
        target = indices[:, 1:].transpose().reshape(-1, )  # (N * num_nbrs,)
        dist = distances[:, 1:].transpose().reshape(-1, )  # (N * num_nbrs,)
        edges = [(src[i], target[i], {'weight': dist[i]}) for i in range(src.shape[0])]
        return src, target, edges

    def create_laplacian(self, rows, cols):
        """
        :return: L, D. both (N, N)
        """
        W = np.zeros((self.num_points, self.num_points))
        W[rows, cols] = 1
        W[cols, rows] = 1
        D = np.diag(np.sum(W, axis=-1))
        L = D - W
        return L, D

    def create_distance_mat(self, edges):
        G = nx.Graph()
        G.add_nodes_from([*range(self.num_points)])
        G.add_edges_from(edges)
        length = dict(nx.all_pairs_dijkstra_path_length(G))
        dist_mat = np.zeros((self.num_points, self.num_points))
        for i in [*range(self.num_points)]:
            dist_mat[i, np.fromiter(length[i].keys(), int)] = np.fromiter(length[i].values(), float)
        return dist_mat

    def get_spectral_dist(self, L, D, dist_mat):
        _, eigen_vec = sp.linalg.eigh(a=L, b=D, eigvals=(1, self.num_eigen))
        return eigen_vec.transpose() @ dist_mat @ eigen_vec


def create_spectral_dataset(train=True, num_eigen=100, num_nbrs=5):
    if train:
        name = 'train'
    else:
        name = 'test'
    bs = 2048
    pc_files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{name}_files.txt')
    ds = CreateSpectralDist(pc_files, num_eigen=num_eigen, num_nbrs=num_nbrs)
    dl = DataLoader(ds, bs, shuffle=False, num_workers=7)

    dl_iter = iter(dl)
    num_batches = len(dl.batch_sampler)
    for batch_idx in range(num_batches):
        print(f'Working on file {batch_idx}', flush=True)
        spectral_dist, label = next(dl_iter)
        print(f'spectral_dist.shape={spectral_dist.shape}')
        print(f'label.shape={label.shape}')
        with h5py.File(f'../data/spectral/spectral_data_{name}{batch_idx}.h5', 'w') as hf:
            hf.create_dataset("spectral_dist",  data=spectral_dist)
            hf.create_dataset("label",  data=label)
    return


class Spectral40Ds(Dataset):
    def __init__(self, h5_files: List[str], size: int):
        super().__init__()
        self.data_name = 'spectral_dist'
        self.size = size
        self.tot_examples = 0
        self.examples = []
        self.train = 'train' in h5_files[0]
        for h5_file in h5_files:
            current_data, current_label = self.load_h5(h5_file)
            current_data = current_data[:, 0:self.size, 0:self.size]
            for i in range(current_data.shape[0]):
                self.examples.append((current_data[i, :, :][np.newaxis, ...], current_label[i, :]))
            self.tot_examples += current_data.shape[0]

    def __getitem__(self, index):
        item, label = self.get_numpy_data(index)
        item_tensor = torch.from_numpy(item).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, label_tensor

    def __len__(self):
        return self.tot_examples

    def get_numpy_data(self, index):
        return self.examples[index]

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename)
        data = f[self.data_name][:]
        label = f['label'][:]
        return data, label


class SpectralSimpleNet(nn.Module):
    def __init__(self, mat_size: int):
        super().__init__()
        modules = []  # (B, 1, S, S)
        modules.extend([nn.Conv2d(1, 16, kernel_size=(5, 5)),  # (B, 16, S-4, S-4)
                        nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 16, S/2-2, S/2-2)
                        nn.ReLU()])
        modules.extend([nn.Conv2d(16, 32, kernel_size=(5, 5)),  # (B, 32, S/2-6, S/2-6)
                        nn.Dropout2d(),
                        nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 32, S/4-3, S/4-3)
                        nn.ReLU()])
        modules.extend([nn.Conv2d(32, 64, kernel_size=(5, 5)),  # (B, 64, S/4-7, S/4-7)
                        nn.Dropout2d(),
                        # nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 64, S/8-7, S/2-7)
                        nn.ReLU()])
        self.feature_seq = nn.Sequential(*modules)
        self.num_features = int(64 * (mat_size / 4 - 7) ** 2)
        modules = []
        modules.extend([nn.Linear(self.num_features, 256),  # (B, 256)
                        nn.ReLU(),
                        nn.Dropout()])
        modules.extend([nn.Linear(256, 128),  # (B, 256)
                        nn.ReLU(),
                        nn.Dropout()])
        modules.append(nn.Linear(128, 40))  # (B, 40)
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        bs = x.shape[0]
        x = self.feature_seq(x).view(bs, self.num_features)
        return self.seq(x)


class SpectralSimpleNetBn(nn.Module):
    def __init__(self, mat_size: int):
        super().__init__()
        modules = []  # (B, 1, S, S)
        modules.extend([nn.Conv2d(1, 16, kernel_size=(5, 5)),  # (B, 16, S-4, S-4)
                        nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 16, S/2-2, S/2-2)
                        nn.ReLU()])
        modules.extend([nn.Conv2d(16, 32, kernel_size=(5, 5)),  # (B, 32, S/2-6, S/2-6)
                        nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 32, S/4-3, S/4-3)
                        nn.ReLU(),
                        nn.BatchNorm2d(32)])
        modules.extend([nn.Conv2d(32, 64, kernel_size=(5, 5)),  # (B, 64, S/4-7, S/4-7)
                        # nn.Dropout2d(),
                        # nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 64, S/8-7, S/2-7)
                        nn.ReLU(),
                        nn.BatchNorm2d(64)])
        self.feature_seq = nn.Sequential(*modules)
        self.num_features = int(64 * (mat_size / 4 - 7) ** 2)
        modules = []
        modules.extend([nn.Linear(self.num_features, 256),  # (B, 256)
                        nn.ReLU(),
                        nn.BatchNorm1d(256)])
        modules.extend([nn.Linear(256, 128),  # (B, 256)
                        nn.ReLU(),
                        nn.BatchNorm1d(128)])
        modules.append(nn.Linear(128, 40))  # (B, 40)
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        bs = x.shape[0]
        x = self.feature_seq(x).view(bs, self.num_features)
        return self.seq(x)


def train_spectral_net(matrix_size):
    bs_train, bs_test = 32, 32
    train_files = get_files_list('../data/spectral/spectral_train_files.txt')
    test_files = get_files_list('../data/spectral/spectral_test_files.txt')

    ds_train = Spectral40Ds(train_files, size=matrix_size)
    ds_test = Spectral40Ds(test_files, size=matrix_size)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True)

    lr = 1e-3
    min_lr = 1e-5
    l2_reg = 0
    our_model = SpectralSimpleNet(mat_size=matrix_size)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
    trainer = NetTrainer(our_model, loss_fn, optimizer, scheduler, min_lr=min_lr)

    expr_name = f'Spectral-t3'
    if os.path.isfile(f'results/{expr_name}.pt'):
        os.remove(f'results/{expr_name}.pt')
    fit_res = trainer.fit(dl_train, dl_test, num_epochs=10000, early_stopping=50, checkpoints=expr_name)
    return


if __name__ == '__main__':
    name = 'train'
    pc_files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{name}_files.txt')
    data_set = CreateSpectralDist(h5_files=pc_files, num_nbrs=5, num_eigen=32)
    import time
    start_time = time.time()
    lst = []
    num_examples = 10
    for ind in [*range(num_examples)]:
        sd, label = data_set.__getitem__(ind)
        lst.append((sd, label))
    print(f'--- {(time.time() - start_time)/num_examples} seconds ---')

    # train_spectral_net(matrix_size=36)

# def load_h5(h5_filename):
#     f = h5py.File(h5_filename)
#     data = f['data'][:]
#     label = f['label'][:]
#     return data, label


# def get_knn(pc):
#     num_nbrs = 5
#     nbrs = NearestNeighbors(n_neighbors=num_nbrs+1, algorithm='auto', metric='euclidean').fit(pc)  # +1 including the point itself
#     distances, indices = nbrs.kneighbors(pc)  # (N, num_nbrs + 1)
#     src = np.tile(indices[:, 0], reps=num_nbrs)  # (N * num_nbrs,)
#     target = indices[:, 1:].transpose().reshape(-1,)  # (N * num_nbrs,)
#     dist = distances[:, 1:].transpose().reshape(-1,)  # (N * num_nbrs,)
#     edges = [(src[i], target[i], {'weight': dist[i]}) for i in range(src.shape[0])]
#     return src, target, edges


# def create_laplacian(N, rows, cols):
#     """
#     :return: L, D. both (N, N)
#     """
#     W = np.zeros((N, N))
#     W[rows, cols] = 1
#     W[cols, rows] = 1
#     D = np.diag(np.sum(W, axis=-1))
#     L = D - W
#     return L, D


# def create_distance_mat(num_points, edges):
#     G = nx.Graph()
#     G.add_nodes_from([*range(num_points)])
#     G.add_edges_from(edges)
#     length = dict(nx.all_pairs_dijkstra_path_length(G))
#     dist_mat = np.zeros((num_points, num_points))
#     for i in [*range(num_points)]:
#         dist_mat[i, np.fromiter(length[i].keys(), int)] = np.fromiter(length[i].values(), float)
#     return dist_mat


# def get_spectral_dist(L, D, dist_mat, num_eigen):
#     _, eigen_vec = sp.linalg.eigh(a=L, b=D, eigvals=(1, num_eigen))
#     return eigen_vec.transpose() @ dist_mat @ eigen_vec


# def spectral_dist_from_pc(ind):
#     num_points = 1024
#     h5_file = '../data/modelnet40_ply_hdf5_2048/ply_data_test1.h5'
#     current_data, current_label = load_h5(h5_file)
#     pc = current_data[ind, 0:num_points, :]
#     label = current_label[ind, :]
#     src, target, edges = get_knn(pc)
#     L, D = create_laplacian(num_points, src, target)
#
#     dist_mat = create_distance_mat(num_points, edges)
#     sd = get_spectral_dist(L, D, dist_mat, num_eigen=100)
#     return sd, label


# import time
# start_time = time.time()
# lst = []
# for ind in [*range(10)]:
#     sd, label = spectral_dist_from_pc(ind)
#     lst.append((sd, label))
    # plt.figure()
    # plt.imshow(sd, extent=[0, 1, 0, 1])
    # plt.colorbar()
    # plt.title(f'{label}')

# print(f'--- {time.time() - start_time} seconds ---')
# plt.show()
# for i in [0,1,2]:
#     print('{}'.format(length[i][0]))
    # print('{}: {}'.format(node, length[2][node]))
# print(*nx.all_pairs_dijkstra_path_length(G))
# print(len(edges))



