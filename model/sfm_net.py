import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from training import NetTrainer, BatchResult
import os
import h5py
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from experiments import get_files_list
from spectral_training import ResNet, BasicBlock


class SfmDs(Dataset):
    def __init__(self, lbo_files: List[str], pc_files: List[str], num_points: int, num_eigen: int, num_nbrs: int):
        """
        :param pc_files:
        :param num_points:
        :param num_nbrs: includes the point itself.
        """
        super().__init__()
        self.num_points = num_points
        self.num_eigen = num_eigen
        self.train = 'train' in pc_files[0]
        self.examples = []
        self.lbo_vecs = []
        for h5_file in lbo_files:
            eig_vecs = self.load_h5(h5_file, 'eigen_vec')
            # eig_vals = self.load_h5(h5_file, 'eigen_val')
            eig_vecs = eig_vecs[:, :, 0:num_eigen]
            # eig_vals = eig_vals[:, 0:num_eigen]
            for i in range(eig_vecs.shape[0]):
                self.lbo_vecs.append((eig_vecs[i, :, :]))

        for h5_file in pc_files:
            labels = self.load_h5(h5_file, 'label')
            data = self.load_h5(h5_file, 'data')
            # data = self.load_h5(h5_file, 'pc')
            # all_normals = self.load_h5(h5_file, 'normals')
            # print(f'normals.shape={normals.shape}')
            # print(f'data.shape={data.shape}')
            # print(f'labels.shape={labels.shape}')
            for i in range(labels.shape[0]):
                label = labels[i, :]
                pc = data[i, 0:self.num_points, :]  # (num_points, 3)
                # normals = all_normals[i, 0:self.num_points, :]  # (num_points, 3)
                # nbrs_idx, nbrs_dist = self.get_knn(pc=pc, num_nbrs=num_nbrs)  # (N*K, ), (N*K, )
                # self.examples.append((np.concatenate((pc.transpose(), normals.transpose()), axis=0), nbrs_idx, nbrs_dist, label))
                self.examples.append((pc.transpose(), label))

    def __getitem__(self, index):
        # pc, nbrs_idx, nbrs_dist, label = self.examples[index]
        pc, label = self.examples[index]
        lbo = self.lbo_vecs[index]
        if self.train:
            # pc = self.jitter_pc(self.rotate_pc(pc))
            pc = self.rotate_pc(pc)
            lbo = self.rand_sign(lbo)
        pc_tensor = torch.from_numpy(pc).float()
        lbo_tensor = torch.from_numpy(lbo).float()
        # nbrs_idx_tensor = torch.from_numpy(nbrs_idx).long()
        # nbrs_dist_tensor = torch.from_numpy(nbrs_dist).float()
        # normals_tensor = torch.from_numpy(normals).float()
        label_tensor = torch.from_numpy(label).long()
        # return pc_tensor, nbrs_idx_tensor, nbrs_dist_tensor, lbo_tensor, label_tensor
        return pc_tensor, lbo_tensor, label_tensor

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def load_h5(h5_filename, data_name):
        f = h5py.File(h5_filename)
        data = f[data_name][:]
        return data

    @staticmethod
    def rotate_pc(pc):
        """
        :param pc: (3, N)
        :return: pc rotated around y axis.
        """
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_y = np.array([[cosval, 0, sinval],
                               [0, 1, 0],
                               [-sinval, 0, cosval]])
        return rotation_y @ pc

    @staticmethod
    def jitter_pc(pc, sigma=0.005, clip=0.02):
        assert (clip > 0)
        noise = np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip)
        return pc + noise

    def rand_sign(self, lbo):
        sign = np.random.choice([-1, 1], size=(1, self.num_eigen))
        return lbo * sign

    @staticmethod
    def get_knn(pc, num_nbrs):
        nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='auto', metric='euclidean').fit(pc)
        distances, indices = nbrs.kneighbors(pc)  # (N, K)
        nbr_idx = indices.reshape(-1, )
        nbr_dist = distances.reshape(1, -1)
        return nbr_idx, nbr_dist


def batched_index_select(x, dim, index):
    views = [x.shape[0]] + [1 if i != dim else -1 for i in range(1, len(x.shape))]
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(x, dim, index)


class PointCloudConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, num_nbs: int, max_pool: bool):
        super().__init__()
        self.avg = not max_pool
        self.fc_x = nn.Conv1d(c_in, c_out, kernel_size=1)
        if max_pool:
            self.pool = nn.MaxPool1d(kernel_size=num_nbs)
        else:
            self.pool = nn.AvgPool1d(kernel_size=num_nbs)
            self.fc_d = nn.Conv1d(1, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x, nbrs_idx, nbrs_dist):
        """
        N - num_points.
        K - num_nbrs.
        :param x: (B, Cin, N) contains C channels descriptor for each vertex.
        :param nbrs_idx: (B, N * K) contains neighbours indices.
        :param nbrs_dist: (B, 1, N * K) contains neighbours distances.
        :return:
        """
        x = self.fc_x(x)  # (B, Cin, N) --> (B, C, N)
        x = batched_index_select(x, 2, nbrs_idx)  # (B, C, N) --> (B, C, NK)
        if self.avg:
            nbrs_dist = torch.exp(-F.relu(self.fc_d(nbrs_dist)))  # (B, 1, NK) --> (B, C, NK)
            x = x * nbrs_dist
        x = self.pool(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class SfmModel(nn.Module):
    def __init__(self, num_eigen: int, num_nbs: int):
        super().__init__()
        modules = []
        filters = (3, num_eigen, num_eigen, num_eigen)
        for c_in, c_out in zip(filters, filters[1:]):
            modules.extend([nn.Conv1d(c_in, c_out, kernel_size=1),
                            nn.BatchNorm1d(c_out),
                            nn.ReLU()])
        self.feat_seq = nn.Sequential(*modules)
        # self.conv1 = PointCloudConv(c_in=6, c_out=64, num_nbs=num_nbs, max_pool=True)
        # self.conv2 = PointCloudConv(c_in=64, c_out=num_eigen, num_nbs=num_nbs, max_pool=True)
        # self.conv3 = PointCloudConv(c_in=num_eigen, c_out=num_eigen, num_nbs=num_nbs, max_pool=True)

        # self.lbo_conv1 = PointCloudConv(c_in=num_eigen, c_out=num_eigen, num_nbs=num_nbs, max_pool=True)
        # self.lbo_conv2 = PointCloudConv(c_in=num_eigen, c_out=num_eigen, num_nbs=num_nbs, max_pool=True)
        # self.lbo_conv3 = PointCloudConv(c_in=num_eigen, c_out=num_eigen, num_nbs=num_nbs, max_pool=True)

        # modules = []
        # filters = (num_eigen, num_eigen, num_eigen)
        # for c_in, c_out in zip(filters, filters[1:]):
        #     modules.extend([nn.Conv1d(c_in, c_out, kernel_size=1),
        #                     nn.BatchNorm1d(c_out),
        #                     nn.ReLU()])
        # self.lbo_seq = nn.Sequential(*modules)

        # modules = []  # (B, 1, S, S)
        # modules.extend([nn.Conv2d(1, 16, kernel_size=(3, 3), padding=True),  # (B, 16, S, S)
        #                 nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 16, S/2, S/2)
        #                 nn.ReLU()])
        # modules.extend([nn.Conv2d(16, 32, kernel_size=(3, 3), padding=True),  # (B, 32, S/2, S/2)
        #                 nn.BatchNorm2d(32),
        #                 nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 32, S/4, S/4)
        #                 nn.ReLU()])
        # modules.extend([nn.Conv2d(32, 64, kernel_size=(3, 3), padding=True),  # (B, 64, S/4, S/4)
        #                 nn.BatchNorm2d(64),
        #                 nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 64, S/8, S/8)
        #                 nn.ReLU()])
        # modules.extend([nn.Conv2d(64, 128, kernel_size=(3, 3), padding=True),  # (B, 128, S/8, S/8)
        #                 nn.BatchNorm2d(128),
        #                 nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 40, S/16, S/16)
        #                 nn.ReLU()])
        # modules.extend([nn.Conv2d(128, 256, kernel_size=(3, 3), padding=True),  # (B, 128, S/16, S/16)
        #                 nn.BatchNorm2d(256),
        #                 nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 128, S/32, S/32)
        #                 nn.ReLU()])
        # self.mat_seq = nn.Sequential(*modules)
        #
        # affine_layers = []
        # channels = (256, 128)
        # for c_in, c_out in zip(channels, channels[1:]):
        #     affine_layers.extend([nn.Linear(c_in, c_out),
        #                          nn.BatchNorm1d(c_out),
        #                          nn.ReLU(),
        #                          nn.Dropout()])
        # affine_layers.append(nn.Linear(channels[-1], 40))
        # self.affine_seq = nn.Sequential(*affine_layers)
        #
        # modules = []
        # modules.extend([nn.Linear(512, 128),  # (B, 256)
        #                 nn.ReLU(),
        #                 nn.Dropout()])
        # modules.append(nn.Linear(128, 40))  # (B, 40)
        # self.affine_seq = nn.Sequential(*modules)
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2])

    # def forward(self, x, nbrs_idx, nbrs_dist, lbo):
    def forward(self, x, lbo):
        """
        : x: (B, 3, N)
        : lbo: (B, N, 64)
        :
        """
        # lam = torch.exp(-lam)
        # lam = lam.view(lam.shape[0], 1, lam.shape[1])
        # lam = self.lambda_seq(lam)  # (B, 64, 1) --> (B, 64, 1)
        # print(f'lam.shape={lam.shape}, lbo.shape={lbo.shape}')
        # lbo = lbo * lam

        # lbo = lbo.transpose(-1, -2)
        # lbo = self.lbo_conv1(lbo, nbrs_idx, nbrs_dist)
        # lbo = self.lbo_conv2(lbo, nbrs_idx, nbrs_dist)
        # lbo = self.lbo_conv3(lbo, nbrs_idx, nbrs_dist)
        # lbo = self.lbo_seq(lbo)
        # lbo = lbo.transpose(-1, -2)

        # x = self.conv1(x, nbrs_idx, nbrs_dist)
        # x = self.conv2(x, nbrs_idx, nbrs_dist)
        # x = self.conv3(x, nbrs_idx, nbrs_dist)
        # x = self.conv4(x, nbrs_idx, nbrs_dist)
        x = self.feat_seq(x)  # (B, 3, N) --> (B, 64, N)
        x = x.bmm(lbo)   # (B, 64, N), (B, N, 64) --> (B, 64, 64)
        # print(f'\nx.shape={x.shape}')
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.resnet(x)
        # x = self.mat_seq(x)
        # x = x.view(x.shape[0], -1)
        # x = self.affine_seq(x)
        return x


class SfmNetTrainer(NetTrainer):
    def __init__(self, model, loss_fn, optimizer, scheduler, min_lr):
        super().__init__(model, loss_fn, optimizer, scheduler, min_lr=min_lr)

    def train_batch(self, batch) -> BatchResult:
        x, lbo, y = batch
        # x, ind, dist, lbo, y = batch
        x, lbo, y = x.to(self.device), lbo.to(self.device), y.view(-1,).to(self.device)
        # x, ind, dist, lbo, y = x.to(self.device), ind.to(self.device), dist.to(self.device), lbo.to(self.device), y.view(-1,).to(self.device)
        self.optimizer.zero_grad()
        # y_pred = self.model(x, ind, dist, lbo)
        y_pred = self.model(x, lbo)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            # x, ind, dist, lbo, y = batch
            # x, ind, dist, lbo, y = x.to(self.device), ind.to(self.device), dist.to(self.device), lbo.to(
            #     self.device), y.view(-1, ).to(self.device)
            x, lbo, y = batch
            x, lbo, y = x.to(self.device), lbo.to(self.device), y.view(-1, ).to(self.device)
            # y_pred = self.model(x, ind, dist, lbo)
            y_pred = self.model(x, lbo)
            loss = self.loss_fn(y_pred, y)
            num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
            return BatchResult(loss.item(), num_correct.item())


def train_net(expr_name: str, num_eigen: int, num_nbrs: int):
    num_points = 1024
    bs_train, bs_test = 32, 32
    train_files_pc = get_files_list(f'../data/modelnet40_ply_hdf5_2048/train_files.txt')
    test_files_pc = get_files_list(f'../data/modelnet40_ply_hdf5_2048/test_files.txt')
    # train_files_pc = get_files_list(f'../data/normals_1024/normals_train_files.txt')
    # test_files_pc = get_files_list(f'../data/normals_1024/normals_test_files.txt')

    train_files_lbo = get_files_list(f'../data/lbo_eig_{num_points}/spectral_train_files.txt')
    test_files_lbo = get_files_list(f'../data/lbo_eig_{num_points}/spectral_test_files.txt')

    ds_train = SfmDs(lbo_files=train_files_lbo, pc_files=train_files_pc, num_points=num_points, num_eigen=num_eigen, num_nbrs=num_nbrs)
    ds_test = SfmDs(lbo_files=test_files_lbo, pc_files=test_files_pc, num_points=num_points, num_eigen=num_eigen, num_nbrs=num_nbrs)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True, num_workers=4)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True, num_workers=4)

    lr = 5e-4
    min_lr = 1e-5
    l2_reg = 0
    model = SfmModel(num_eigen=num_eigen, num_nbs=num_nbrs)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    trainer = SfmNetTrainer(model, loss_fn, optimizer, scheduler, min_lr=min_lr)

    if os.path.isfile(f'results/{expr_name}.pt'):
        os.remove(f'results/{expr_name}.pt')
    _ = trainer.fit(dl_train, dl_test, num_epochs=400, early_stopping=50, checkpoints=expr_name)
    return


if __name__ == '__main__':
    train_net(expr_name='sfm-net-t1', num_eigen=64, num_nbrs=8)