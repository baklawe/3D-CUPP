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


if __name__ == '__main__':
    create_spectral_dataset(train=True)
    create_spectral_dataset(train=False)

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



