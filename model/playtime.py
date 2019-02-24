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



def plot_pc(pc, c=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = pc[:, 0], pc[:, 1], pc[:, 2]
    if c is not None:
        ax.scatter(xs, ys, zs, c=c)
    else:
        ax.scatter(xs, ys, zs)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    # plt.show()


def get_item(examples, index):
    pc, label = examples[index]
    src, target, wt = get_knn(pc=pc, num_nbrs=5)

    L, D = create_laplacian(num_points=1024, rows=src, cols=target)
    _, eigen_vec = sp.linalg.eigh(a=L, b=D, eigvals=(1, 20))
    # edges = [(src[i], target[i], {'weight': wt[i]}) for i in range(src.shape[0])]
    # dist_mat1 = create_distance_mat(num_points=1024, edges=edges)
    dist_mat = distance_mat.get_distance_m(1024, src, target, wt)
    dist_mat[dist_mat == np.inf] = 0
    # if np.sum(np.isinf(dist_mat)) > 0.0:
    #     print(f'np.sum(np.isinf(dist_mat))={np.sum(np.isinf(dist_mat))}')
    #     print(f'np.sum(np.isnan(dist_mat))={np.sum(np.isnan(dist_mat))}')
    #     print(f'isnan={np.argwhere(np.isnan(dist_mat))}')
    #     print(f'isinf={np.argwhere(np.isinf(dist_mat))}')
    #     print(f'dist_mat=\n{dist_mat}')
    #     print(f'index={index}\n\n')
    #     print(f'label={label}\n\n')
    # col = np.argwhere(np.isinf(dist_mat))[0, 1]
    plot_pc(pc, eigen_vec[:, 3])
        # plot_pc(pc, dist_mat[:, col])
        # plot_pc(pc)
    spectral_dist = get_spectral_dist(num_eigen=32, L=L, D=D, dist_mat=dist_mat)
    return spectral_dist, label


def get_knn(pc, num_nbrs):
    # +1 including the point itself
    nbrs = NearestNeighbors(n_neighbors=num_nbrs + 1, algorithm='auto', metric='euclidean').fit(pc)
    distances, indices = nbrs.kneighbors(pc)  # (N, num_nbrs + 1)
    src = np.tile(indices[:, 0], reps=num_nbrs)  # (N * num_nbrs,)
    target = indices[:, 1:].transpose().reshape(-1, )  # (N * num_nbrs,)
    wt = distances[:, 1:].transpose().reshape(-1, )  # (N * num_nbrs,)
    # edges = [(src[i], target[i], {'weight': wt[i]}) for i in range(src.shape[0])]
    return src, target, wt


def create_laplacian(num_points, rows, cols):
    """
    :return: L, D. both (N, N)
    """
    W = np.zeros((num_points, num_points))
    W[rows, cols] = 1
    W[cols, rows] = 1
    D = np.diag(np.sum(W, axis=-1))
    L = D - W
    return L, D


def create_distance_mat(num_points, edges):
    G = nx.Graph()
    G.add_nodes_from([*range(num_points)])
    G.add_edges_from(edges)
    length = dict(nx.all_pairs_dijkstra_path_length(G))
    dist_mat = np.zeros((num_points, num_points))
    for i in [*range(num_points)]:
        dist_mat[i, np.fromiter(length[i].keys(), int)] = np.fromiter(length[i].values(), float)
    return dist_mat


def get_spectral_dist(num_eigen, L, D, dist_mat):
    _, eigen_vec = sp.linalg.eigh(a=L, b=D, eigvals=(1, num_eigen))
    return eigen_vec.transpose() @ dist_mat @ eigen_vec


def load_spectral(h5_filename):
    f = h5py.File(h5_filename)
    data = f['spectral_dist'][:]
    label = f['label'][:]
    return data, label


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def time_check():
    lp = LineProfiler()
    lp.add_function(create_distance_mat)  # add additional function to profile
    lp_wrapper = lp(get_item)
    num_points = 1024
    tot_examples = 0
    examples = []
    train = 'train'
    h5_files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{train}_files.txt')
    for h5_file in h5_files:
        current_data, current_label = load_h5(h5_file)
        current_data = current_data[:, 0:num_points, :]
        for i in range(current_data.shape[0]):
            examples.append((current_data[i, :, :], current_label[i, :]))
        tot_examples += current_data.shape[0]
    sd, label = lp_wrapper(examples=examples, index=83)
    lp.print_stats()
    return


def create_laplacian_heat_kernel(num_points, rows, cols, wt, t):
    """
    :return: L, D. both (N, N)
    """
    W = np.zeros((num_points, num_points))

    if t != 0:
        heat_w = np.exp(-(wt ** 2) / (4 * t))
    else:
        heat_w = 1
    W[rows, cols] = heat_w
    W[cols, rows] = heat_w
    D = np.diag(np.sum(W, axis=-1))
    L = D - W
    return L, D


# num_points = 1024
# h5_files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/train_files.txt')
# current_data, current_label = load_h5(h5_files[0])
#
# ind = 20
# pc = current_data[ind, 0:num_points, :]
# src, target, wt = get_knn(pc=pc, num_nbrs=5)
# num_eigen = 4
# for t in [0, 5e-4]:
#     L, D = create_laplacian_heat_kernel(num_points, src, target, wt, t)
#     eigen_val, eigen_vec = sp.linalg.eigh(a=L, b=D, eigvals=(1, num_eigen))
#
#     # fig, axes = plt.subplots(nrows=num_eigen, ncols=1, figsize=(16, 10), sharex='col')
#     for j in range(num_eigen):
#         plot_pc(pc, eigen_vec[:, j], f'V{j}, lambda={eigen_val[j]} t={t}')
    # vi = 2

# plt.show()
# kdt = sp.spatial.cKDTree(pc)
# num_nbrs = kdt.count_neighbors(kdt, r=0.07, p=2)
# pairs = kdt.query_pairs(r=0.07, p=2)
# print(pairs)


data1, label1 = load_spectral('../data/heat/spectral_data_test0.h5')
data2, label2 = load_spectral('../data/backup/spectral_data_test0.h5')
# # print(label2)
# # print(label1)
print(f'label equal = {np.array_equal(label1, label2)}')
size = 32
for i in range(5):
    mat1 = data1[i, 0:size, 0:size]
    mat2 = data2[i, 0:size, 0:size]
    # mat1[3, 6] = np.inf
    plt.figure(i)
    # plt.imshow(mat1, extent=[0, 1, 0, 1])
    plt.imshow(np.abs(mat1)-np.abs(mat2), extent=[0, 1, 0, 1])
    plt.colorbar()
    # plt.title(f'label')
plt.show()
# if __name__ == '__main__':
    # lp = LineProfiler()
    # lp_wrapper = lp(get_item)
    # # lp_wrapper = lp(create_distance_mat)
    # # lp_wrapper = lp(get_spectral_dist)
    # # lp_wrapper = lp(get_spectral_dist)
    # # lp_wrapper(numbers)
    # # lp.print_stats()
    #
    # num_points = 1024
    # tot_examples = 0
    # examples = []
    # lp_wrapper(examples)
    # train = 'train'
    # h5_files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{train}_files.txt')
    # for h5_file in h5_files:
    #     current_data, current_label = load_h5(h5_file)
    #     current_data = current_data[:, 0:num_points, :]
    #     for i in range(current_data.shape[0]):
    #         examples.append((current_data[i, :, :], current_label[i, :]))
    #     tot_examples += current_data.shape[0]
    #
    # start_time = time.time()
    # lst = []
    # num_examples = 10
    # for ind in [*range(num_examples)]:
    #     # sd, label = get_item(examples=examples, index=ind)
    #     sd, label = lp_wrapper(examples=examples, index=ind)
    #     lst.append((sd, label))
    # print(f'--- {(time.time() - start_time)/num_examples} seconds ---')
    # lp.print_stats()


