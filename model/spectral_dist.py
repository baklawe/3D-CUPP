import h5py
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy as sp
from laplacian import ModelNet40Base
from typing import List
from experiments import get_files_list
from torch.utils.data import DataLoader
import distance_mat
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


class CreateSpectralDist(ModelNet40Base):
    """
    This class helps to create the eigenfunction & eigenvalues of the LBO
    """
    def __init__(self, h5_files: List[str], num_nbrs: int, num_eigen: int):
        super().__init__(h5_files=h5_files)
        self.num_nbrs = num_nbrs
        self.num_eigen = num_eigen
        self.num_points = 1024
        self.t = 1e-2

    def __getitem__(self, index):
        """
        :return: eigenfunctions & eigenvalues associated with the point cloud.
        """
        pc, label = self.get_pc(index)
        pc = pc[0:self.num_points, :]
        src, target, wt = self.get_knn(pc)
        src, target, wt = self.connect_graph(pc, src, target, wt)
        L, D = self.create_laplacian(src, target, wt)
        dist_mat = distance_mat.get_distance_m(self.num_points, src, target, wt)
        spectral_dist, eigen_val = self.get_spectral_dist(L, D, dist_mat)
        print(f'Got index {index}')
        return spectral_dist, eigen_val, label

    def __len__(self):
        return self.tot_examples

    def get_knn(self, pc):
        # +1 including the point itself
        nbrs = NearestNeighbors(n_neighbors=self.num_nbrs + 1, algorithm='auto', metric='euclidean').fit(pc)
        distances, indices = nbrs.kneighbors(pc)  # (N, num_nbrs + 1)
        src = np.tile(indices[:, 0], reps=self.num_nbrs)  # (N * num_nbrs,)
        target = indices[:, 1:].transpose().reshape(-1, )  # (N * num_nbrs,)
        wt = distances[:, 1:].transpose().reshape(-1, )  # (N * num_nbrs,)
        return src, target, wt

    def connect_graph(self, pc, src, target, wt):
        W = np.zeros((self.num_points, self.num_points))
        W[src, target] = 1
        n_components, labels = connected_components(csgraph=W, directed=False, return_labels=True)
        while n_components is not 1:
            part1 = pc[labels == 0, :]
            part2 = pc[labels != 0, :]
            y = cdist(part1, part2, 'euclidean')
            y_flat = y.ravel()
            idx = np.argsort(y_flat)[0:5:1]
            new_wt = y_flat[idx]
            idx = np.column_stack(np.unravel_index(idx, y.shape))
            new_src = np.argwhere(labels == 0)[idx[:, 0]]
            new_trg = np.argwhere(labels != 0)[idx[:, 1]]
            W[new_src, new_trg] = 1
            src = np.concatenate((src.reshape(-1, ), new_src.reshape(-1, )), axis=0)
            target = np.concatenate((target.reshape(-1, ), new_trg.reshape(-1, )), axis=0)
            wt = np.concatenate((wt.reshape(-1, ), new_wt.reshape(-1, )), axis=0)
            n_components, labels = connected_components(csgraph=W, directed=False, return_labels=True)
        return src, target, wt

    def create_laplacian(self, rows, cols, wt):
        """
        We want the laplacian with the weights:
        exp(-|x-y|^2/4t) if |x-y|< eps, 0 else.
        see http://web.cse.ohio-state.edu/~belkin.8/papers/LEM_NC_03.pdf
        Another thing: we want 1 connected-component.
        :return: L, D. both (N, N)
        """
        W = np.zeros((self.num_points, self.num_points))
        if self.t != 0:
            heat_w = np.exp(-(wt ** 2) / (4 * self.t))
        else:
            heat_w = 1
        W[rows, cols] = heat_w
        W[cols, rows] = heat_w
        D = np.diag(np.sum(W, axis=-1))
        L = D - W
        return L, D

    def get_spectral_dist(self, L, D, dist_mat):
        eigen_val, eigen_vec = sp.linalg.eigh(a=L, b=D, eigvals=(1, self.num_eigen))
        spectral_dist = eigen_vec.transpose() @ dist_mat @ eigen_vec
        return spectral_dist, eigen_val


def create_spectral_dataset(train, num_eigen, num_nbrs, num_workers):
    if train:
        name = 'train'
    else:
        name = 'test'
    bs = 512
    dest_dir = 'heat'
    pc_files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{name}_files.txt')
    ds = CreateSpectralDist(pc_files, num_eigen=num_eigen, num_nbrs=num_nbrs)
    dl = DataLoader(ds, bs, shuffle=False, num_workers=num_workers)

    files_list = []
    dl_iter = iter(dl)
    num_batches = len(dl.batch_sampler)
    for batch_idx in range(num_batches):
        print(f'Working on file {batch_idx}', flush=True)
        spectral_dist, eigen_val, label = next(dl_iter)
        print(f'spectral_dist.shape={spectral_dist.shape}')
        print(f'eigen_val.shape={eigen_val.shape}')
        print(f'label.shape={label.shape}')
        file_name = f'data/{dest_dir}/spectral_data_{name}{batch_idx}.h5'
        files_list.append(file_name)
        with h5py.File('../' + file_name, 'w') as hf:
            hf.create_dataset("spectral_dist", data=spectral_dist)
            hf.create_dataset("eigen_val", data=eigen_val)
            hf.create_dataset("label", data=label)
    txt_name = f'../data/{dest_dir}/spectral_{name}_files.txt'
    with open(txt_name, 'w') as f:
        f.write('\n'.join(files_list))
    return


if __name__ == '__main__':
    create_spectral_dataset(train=True, num_eigen=64, num_nbrs=10, num_workers=8)
    create_spectral_dataset(train=False, num_eigen=64, num_nbrs=10, num_workers=8)


