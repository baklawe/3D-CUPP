import scipy as sp
from training import ModelNet40Ds, NetTrainer
from model import View1D
from experiments import get_files_list
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import h5py
from torch.utils.data import DataLoader
from typing import List
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
import inspect
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# from utils.normals_Hough.python.lib.python import NormalEstimatorHough as NormalEstimator


class ModelNet40Base(Dataset):
    def __init__(self, h5_files: List[str]):
        super().__init__()
        self.num_points = 1024
        self.tot_examples = 0
        self.examples = []
        self.train = 'train' in h5_files[0]
        for h5_file in h5_files:
            current_data, current_label = self.load_h5(h5_file)
            current_data = current_data[:, 0:self.num_points, :]
            for i in range(current_data.shape[0]):
                self.examples.append((current_data[i, :, :], current_label[i, :]))
            self.tot_examples += current_data.shape[0]

    def __getitem__(self, index):
        return self.get_pc(index)

    def __len__(self):
        return self.tot_examples

    def get_pc(self, index):
        return self.examples[index]

    @staticmethod
    def load_h5(h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return data, label


class CreateEigenLBO(ModelNet40Base):
    """
    This class helps to create the eigenfunction & eigenvalues of the LBO
    """
    def __init__(self, h5_files: List[str], num_nbrs: int, num_eigen: int):
        super().__init__(h5_files=h5_files)
        self.num_nbrs = num_nbrs
        self.num_eigen = num_eigen

    def __getitem__(self, index):
        """
        :return: eigenfunctions & eigenvalues associated with the point cloud.
        """
        pc, label = self.get_pc(index)
        eigen_val, eigen_vec = self.calc_eigs(*self.create_laplacian(pc))
        return eigen_val, eigen_vec, label

    def __len__(self):
        return self.tot_examples

    def get_knn(self, pc):
        """
        :param pc: (N, 3)
        :return:
        """
        nbrs = NearestNeighbors(n_neighbors=self.num_nbrs+1, algorithm='auto').fit(pc)  # +1 including the point itself
        _, indices = nbrs.kneighbors(pc)  # (N, self.num_nbrs + 1)
        rows = np.tile(indices[:, 0], reps=self.num_nbrs)  # (N * self.num_nbrs,)
        cols = indices[:, 1:].transpose().reshape(-1,)  # (N * self.num_nbrs,)
        return rows, cols

    def create_laplacian(self, pc):
        """
        :param pc: (N, 3)
        :return: L, D. both (N, N)
        """
        W = np.zeros((pc.shape[0], pc.shape[0]))
        rows, cols = self.get_knn(pc)
        W[rows, cols] = 1
        W[cols, rows] = 1
        D = np.diag(np.sum(W, axis=-1))
        L = D - W
        return L, D

    def calc_eigs(self, L, D):
        return sp.linalg.eigh(a=L, b=D, eigvals=(0, self.num_eigen-1))


def create_eigen_dataset(train=True, num_eigen=100, num_nbrs=5):
    if train:
        name = 'train'
    else:
        name = 'test'
    bs = 2048
    pc_files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{name}_files.txt')
    ds = CreateEigenLBO(pc_files, num_eigen=num_eigen, num_nbrs=num_nbrs)
    dl = DataLoader(ds, bs, shuffle=False, num_workers=20)

    dl_iter = iter(dl)
    num_batches = len(dl.batch_sampler)
    for batch_idx in range(num_batches):
        print(f'Working on file {batch_idx}', flush=True)
        eigval, eigvec, label = next(dl_iter)
        print(f'eigval.shape={eigval.shape}')
        print(f'eigvec.shape={eigvec.shape}')
        print(f'label.shape={label.shape}')
        with h5py.File(f'../data/eigen/eigen_data_{name}{batch_idx}.h5', 'w') as hf:
            hf.create_dataset("eigval",  data=eigval)
            hf.create_dataset("eigvec",  data=eigvec)
            hf.create_dataset("label",  data=label)
    return


class CreateNormals40Ds(ModelNet40Base):
    def __init__(self, h5_files: List[str]):
        super().__init__(h5_files)
        self.estimator = NormalEstimator.NormalEstimatorHough()
        self.estimator.set_K(5)

    def __getitem__(self, index):
        """
        :param index: point cloud index
        :return: Normals (N, 3)
        """
        pc, _ = self.get_pc(index)
        normals = self.calc_normals(pc)
        return normals

    def calc_normals(self, pc):
        pc = np.float64(pc)
        self.estimator.set_points(pc)
        self.estimator.estimate_normals()
        return np.float32(self.estimator.get_normals())


def create_normals_dataset(train=True):
    if train:
        name = 'train'
    else:
        name = 'test'
    bs = 2048
    files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{name}_files.txt')
    ds = CreateNormals40Ds(files)
    dl = DataLoader(ds, bs, shuffle=False, num_workers=20)

    dl_iter = iter(dl)
    num_batches = len(dl.batch_sampler)
    for batch_idx in range(num_batches):
        print(f'Working on file {batch_idx}')
        normal = next(dl_iter)
        print(f'normals.shape={normal.shape}')
        with h5py.File(f'../data/eigen/normal_data_{name}{batch_idx}.h5', 'w') as hf:
            hf.create_dataset("normal",  data=normal)
    return


class EigenNet40Ds(Dataset):
    def __init__(self, h5_files: List[str], evec, c_in=100):
        super().__init__()
        self.num_points = 1024
        self.tot_examples = 0
        self.examples_per_file = 2048
        self.examples = []
        for h5_file in h5_files:
            curr_eigval, curr_eigvec, curr_label = self.load_h5(h5_file)
            self.examples.append((curr_eigval, curr_eigvec, curr_label))
            self.tot_examples += curr_eigval.shape[0]
        self.train = 'train' in h5_files[0]
        self.evec = evec
        self.c_in = c_in

    def __getitem__(self, index):
        item, label = self.get_np_data(index)
        item_tensor = torch.from_numpy(item).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, label_tensor

    def __len__(self):
        return self.tot_examples

    def get_np_data(self, index):
        # Get file name
        file_ind = index // self.examples_per_file
        example_ind = index % self.examples_per_file
        if self.evec:
            _, eigvec, label = self.examples[file_ind]
            eigvec, label = eigvec[example_ind, :, :self.c_in].transpose(), label[example_ind, :]
            assert eigvec.shape == (self.c_in, self.num_points), f'label.shape={eigvec.shape}'
            item = eigvec
        else:
            eigval, _, label = self.examples[file_ind]
            eigval, label = eigval[example_ind, 1:self.c_in+1], label[example_ind, :]
            assert eigval.shape == (self.c_in,), f'item.shape={eigval.shape}'
            item = eigval
        assert label.shape == (1,), f'label.shape={label.shape}'
        return item, label

    @staticmethod
    def load_h5(h5_filename):
        with h5py.File(h5_filename, 'r') as hf:
            eigval = hf['eigval'][:]
            eigvec = hf['eigvec'][:]
            label = hf['label'][:]
        return eigval, eigvec, label


class HearShapeNet(nn.Module):
    def __init__(self, c_in: int):
        """
        Input dim: (B, 1, c_in) Shape spectrum.
        output dim: (B, 40).
        """
        super().__init__()
        affine_layers = []
        # for c_in, c_out in zip(channels, channels[1:]):
        affine_layers.extend([nn.Linear(c_in, 128),
                              nn.ReLU()])
        affine_layers.extend([nn.Linear(128, 128),
                              nn.ReLU(),
                              nn.BatchNorm1d(128)])
        affine_layers.extend([nn.Linear(128, 128),
                              nn.ReLU(),
                              nn.BatchNorm1d(128)])
        affine_layers.append(nn.Linear(128, 40))
        self.affine_seq = nn.Sequential(*affine_layers)

    def forward(self, x):
        return self.affine_seq(x)


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
    plt.show()


def train_shape_net(c_in=30, evec=False):
    bs_train, bs_test = 32, 32
    train_files = get_files_list('../data/eigen/eigen_train_files.txt')
    test_files = get_files_list('../data/eigen/eigen_test_files.txt')

    ds_train = EigenNet40Ds(train_files, evec=evec, c_in=c_in)
    ds_test = EigenNet40Ds(test_files, evec=evec, c_in=c_in)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True)

    lr = 1e-4
    l2_reg = 0
    our_model = HearShapeNet(c_in=c_in)
    # our_model = SpectralPointNet()
    loss_fn = F.cross_entropy  # This criterion combines log_softmax and nll_loss in a single function
    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    trainer = NetTrainer(our_model, loss_fn, optimizer, scheduler)

    expr_name = f'Eigen-t3'
    if os.path.isfile(f'results/{expr_name}.pt'):
        os.remove(f'results/{expr_name}.pt')
    fit_res = trainer.fit(dl_train, dl_test, num_epochs=10000, early_stopping=50, checkpoints=expr_name)
    return


class SpectralPointNet(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N) represents batch of N (usually N=1024) (x, y, z).
        After Conv1d filters: (B, 1024, N).
        After pooling: (B, 1024, 1).
        After View: (B, 1024).
        Linear layers: (B, 1024) -> (B, num_classes). num_classes = 40 for ModelNet40 data-set.
        """
        super().__init__()
        modules = []
        filters = (100, 256, 256, 1024)
        affine = (filters[-1], 512, 256, 40)
        for in_filters, c_out in zip(filters, filters[1:]):
            modules.extend([nn.Conv1d(in_filters, c_out, kernel_size=1),
                            nn.ReLU(),
                            nn.BatchNorm1d(c_out)])
        modules.append(nn.AvgPool1d(kernel_size=1024))
        modules.append(View1D())
        for in_affine, out_affine in zip(affine[:-1], affine[1:-1]):
            modules.extend([nn.Linear(in_affine, out_affine),
                           nn.ReLU(),
                           nn.BatchNorm1d(out_affine)])
        modules.append(nn.Dropout())
        modules.append(nn.Linear(affine[-2], affine[-1]))
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


if __name__ == '__main__':
    train_shape_net(c_in=30, evec=False)
    # target = 'train'
    # files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{target}_files.txt')
    # ds = CreateLBOeigen(h5_files=files, num_nbrs=5, num_eigen=150)
    # ind = 5
    # pc = ds.get_pc(ind)
    # e_val, e_vec = ds.calc_eigs(*ds.create_laplacian(pc))
    # for i in range(10):
    #     plot_pc(pc, e_vec[:, i], title=f'V{i}')
    # plt.show()
    # train_shape_net(c_in=30, evec=False)
    # create_eigen_dataset(train=True, num_eigen=100, num_nbrs=5)
    # create_eigen_dataset(train=False, num_eigen=100, num_nbrs=5)
    # create_normals_dataset(train=True)
    # create_normals_dataset(train=False)

    # target = 'train'
    # bs = 2048
    # files = get_files_list(f'../data/modelnet40_ply_hdf5_2048/{target}_files.txt')
    # t = 100
    # ds = CreateShapeNet40Ds(files, num_evals=100, num_nbrs=5, t=t)
    # ind = 5
    # pc, label = ds.get_np_pc(ind)
    # # print(pc.shape)
    # print(ds.calc_normals(np.float64(pc.transpose())).dtype)
    # e_val, e_vec = ds.calc_eigs(*ds.create_laplacian(pc.transpose()))
    # print(e_val)
    # for i in range(10):
    #     plot_pc(pc.transpose(), e_vec[:, i], title=f'V{i}, t={t}')
    # plt.show()





