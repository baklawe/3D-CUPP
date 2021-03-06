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
from random import randint


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
                self.examples.append((np.expand_dims(current_data[i, :, :], axis=0), current_label[i, :]))
            self.tot_examples += current_data.shape[0]

    def __getitem__(self, index):
        item, label = self.get_numpy_data(index)
        if self.train:
            item = self.rand_sign(item)
            item = self.add_noise(item)
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

    def rand_sign(self, mat):
        sign = np.random.choice([-1, 1], size=self.size)
        sign = np.reshape(sign, (self.size, 1))
        mat = np.multiply(sign.transpose(), mat)
        mat = np.multiply(mat, sign)
        return mat

    def add_noise(self, mat):
        noise = np.random.normal(1, 0.002, (self.size, self.size))
        return mat * noise


class Gil40Ds(Dataset):
    def __init__(self, h5_files: List[str], size: int):
        super().__init__()
        self.data_name = 'gil_dist'
        self.size = size
        self.tot_examples = 0
        self.examples = []
        self.train = 'train' in h5_files[0]
        for h5_file in h5_files:
            current_data, current_label = self.load_h5(h5_file)
            current_data = current_data[:, 0:self.size, 0:self.size]
            for i in range(current_data.shape[0]):
                # self.examples.append((np.expand_dims(current_data[i, :], axis=0), current_label[i, :]))
                self.examples.append((np.diag(current_data[i, :, :]), current_label[i, :]))
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


class SpectralWithEig40Ds(Spectral40Ds):
    def __init__(self, h5_files: List[str], size: int):
        super().__init__(h5_files=h5_files, size=size)
        self.examples = []
        self.tot_examples = 0
        for h5_file in h5_files:
            current_data, current_eig, current_label = self.load_h5_eig(h5_file)
            current_data = current_data[:, 0:self.size, 0:self.size]
            current_eig = current_eig[:, 0:self.size]
            for i in range(current_data.shape[0]):
                self.examples.append((np.expand_dims(current_data[i, :, :], axis=0),
                                      np.expand_dims(current_eig[i, :], axis=0),
                                      current_label[i, :]))
            self.tot_examples += current_data.shape[0]

    def __getitem__(self, index):
        item, eig, label = self.get_numpy_data(index)
        if self.train:
            item = self.rand_sign(item)
            # item = self.add_noise(item)
        item_tensor = torch.from_numpy(item).float()
        eig_tensor = torch.from_numpy(eig).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, eig_tensor, label_tensor

    def get_numpy_data(self, index):
        return self.examples[index]

    def load_h5_eig(self, h5_filename):
        f = h5py.File(h5_filename)
        data = f[self.data_name][:]
        eig = f['eigen_val'][:]
        label = f['label'][:]
        return data, eig, label


class HybMat40Ds(Dataset):
    def __init__(self, lbo_files: List[str], dist_files: List[str], pc_files: List[str], size: int):
        super().__init__()
        self.size = size
        lbo_vecs = []
        lbo_vals = []
        dist_vecs = []
        dist_vals = []
        self.mat_examples = []
        self.label_examples = []
        self.train = 'train' in lbo_files[0]
        for h5_file in lbo_files:
            eig_vecs = self.load_h5(h5_file, 'eigen_vec')
            eig_vals = self.load_h5(h5_file, 'eigen_val')
            eig_vecs = eig_vecs[:, :, 0:self.size]
            eig_vals = eig_vals[:, 0:self.size]
            for i in range(eig_vecs.shape[0]):
                lbo_vecs.append(eig_vecs[i, :, :])
                lbo_vals.append(eig_vals[i, :])
        for h5_file in dist_files:
            eig_vecs = self.load_h5(h5_file, 'eigen_vec')
            eig_vals = self.load_h5(h5_file, 'eigen_val')
            eig_vecs = eig_vecs[:, :, 0:self.size]
            eig_vals = eig_vals[:, 0:self.size]
            for i in range(eig_vecs.shape[0]):
                dist_vecs.append(eig_vecs[i, :, :])
                dist_vals.append(eig_vals[i, :])
        assert (len(lbo_vecs) == len(dist_vecs))
        for lbo_vec, dist_vec, lbo_val, dist_val in zip(lbo_vecs, dist_vecs, lbo_vals, dist_vals):
            # mat = (lbo_vec.transpose() @ dist_vec) @ np.diag(dist_val) @ (dist_vec.transpose() @ lbo_vec)
            #lhs = np.diag(np.exp(-lbo_val)) @ lbo_vec.transpose() @ dist_vec
            #mat = lhs @ np.diag(dist_val) @ lhs.transpose()
            #mat = np.diag(np.exp(-lbo_val)) @ lbo_vec.transpose() @ dist_vec @ np.diag(dist_val)
            mat = lbo_vec.transpose() @ dist_vec @ np.diag(dist_val)
            assert (mat.shape == (self.size, self.size)), f'mat.shape={mat.shape}'
            self.mat_examples.append(mat)
        del lbo_vecs
        del dist_vecs
        del lbo_vals
        del dist_vals

        for h5_file in pc_files:
            current_data = self.load_h5(h5_file, 'label')
            for i in range(current_data.shape[0]):
                self.label_examples.append(current_data[i, :])
        assert (len(self.mat_examples) == len(self.label_examples))

    def __getitem__(self, index):
        item, label = self.get_numpy_data(index)
        if self.train:
            item = self.rand_sign(item)
            #if randint(1, 100) < 4:
                #label = np.mod(label + 1, 40)
                #label = np.array([randint(0, 39)])
            # item = self.add_noise(item)
        item, label = np.expand_dims(item, axis=0), np.expand_dims(label, axis=0)
        item_tensor = torch.from_numpy(item).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, label_tensor

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
        mat = np.diag(sign1) @ mat @ np.diag(sign2)
        return mat

    def rand_sign_1(self, mat):
        sign1 = np.random.choice([-1, 1], size=self.size)
        mat = np.diag(sign1) @ mat @ np.diag(sign1).transpose()
        return mat

    def add_noise(self, mat):
        noise = np.random.normal(1, 0.0001, (self.size, self.size))
        return mat * noise


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=40, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(512 * block.expansion + 384, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # conv1d layers
        eig_layers = []  # (B, 1, 32)
        eig_layers.extend([nn.Conv1d(1, 16, kernel_size=3),  # (B, 16, 30)
                           nn.ReLU()])
        eig_layers.extend([nn.Conv1d(16, 32, kernel_size=3),  # (B, 32, 28)
                           nn.MaxPool1d(kernel_size=2),  # (B, 32, 14)
                           nn.ReLU()])
        eig_layers.extend([nn.Conv1d(32, 64, kernel_size=3),  # (B, 64, 12)
                           nn.MaxPool1d(kernel_size=2),  # (B, 64, 6)
                           nn.ReLU()])
        self.eig_seq = nn.Sequential(*eig_layers)
        return

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_eig(self, x, eig):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        eig = self.eig_seq(eig).view(eig.shape[0], -1)

        x = torch.cat([x, eig], dim=-1)

        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class SpectralDiag(Spectral40Ds):
    def __init__(self, h5_files: List[str], size: int):
        super().__init__(h5_files=h5_files, size=size)

    def __getitem__(self, index):
        item, label = self.get_numpy_data(index)  # (1, 32, 32)
        item = np.diag(item[0, :, :])  # (32,)
        item_tensor = torch.from_numpy(item).float()
        label_tensor = torch.from_numpy(label).long()
        return item_tensor, label_tensor


class SpectralFcNet(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        modules = []
        filters = (c_in, 128, 128)
        for i, (c_in, c_out) in enumerate(zip(filters, filters[1:]), start=1):
            modules.extend([nn.Linear(c_in, c_out), nn.ReLU()])
            if i is not 1:
                modules.append(nn.BatchNorm1d(c_out))
        modules.append(nn.Linear(filters[-1], 40))  # (B, 40)
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        # print(f'x.shape={x.shape}')
        return self.seq(x)


class SpectralConv1dNet(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        modules = []  # (B, 1, 128)
        modules.extend([nn.Conv1d(1, 16, kernel_size=3),  # (B, 16, 30)
                        # nn.MaxPool1d(kernel_size=2),  # (B, 16, 15)
                        nn.ReLU()])
        modules.extend([nn.Conv1d(16, 32, kernel_size=3),  # (B, 32, 28)
                        nn.MaxPool1d(kernel_size=2),  # (B, 32, 14)
                        nn.ReLU()])
        modules.extend([nn.Conv1d(32, 64, kernel_size=3),  # (B, 64, 12)
                        nn.ReLU()])
        self.conv_seq = nn.Sequential(*modules)
        modules = []
        modules.append(nn.Linear(64 * 12, 40))  # (B, 40)
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.conv_seq(x)
        # print(f'x.shape={x.shape}')
        x = x.view(x.shape[0], -1)
        return self.seq(x)


class SpectralSimpleNetBn(nn.Module):
    def __init__(self, mat_size: int):
        super().__init__()
        modules = []  # (B, 1, S, S)
        modules.extend([nn.Conv2d(1, 10, kernel_size=(5, 5)),  # (B, 16, S-4, S-4)
                        nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 16, S/2-2, S/2-2)
                        nn.ReLU()])
        modules.extend([nn.Conv2d(10, 32, kernel_size=(5, 5)),  # (B, 32, S/2-6, S/2-6)
                        nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 32, S/4-3, S/4-3)
                        nn.ReLU(),
                        nn.BatchNorm2d(32)])
        modules.extend([nn.Conv2d(32, 64, kernel_size=(3, 3)),  # (B, 64, S/4-5, S/4-5)
                        # nn.Dropout2d(),
                        # nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 64, S/8-1, S/8-1)
                        nn.ReLU(),
                        nn.BatchNorm2d(64)])
        self.feature_seq = nn.Sequential(*modules)
        self.num_features = int(64 * (mat_size / 4 - 5) ** 2)
        modules = []
        modules.extend([nn.Linear(self.num_features, 128),  # (B, 256)
                        nn.ReLU(),
                        nn.BatchNorm1d(128)])
        # modules.extend([nn.Linear(256, 128),  # (B, 256)
        #                 nn.ReLU(),
        #                 nn.BatchNorm1d(128)])
        modules.append(nn.Linear(128, 40))  # (B, 40)
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        bs = x.shape[0]
        x = self.feature_seq(x)
        x = x.view(bs, self.num_features)
        x = self.seq(x)
        return x


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
        x = self.seq(x)
        if torch.sum(torch.isnan(x)).item() != 0:
            print(x)
        # print('torch.sum(torch.isinf(y_pred))', ))
        # print('torch.sum(torch.isnan(y_pred))', torch.sum(torch.isnan(y_pred)))

        return F.log_softmax(x, dim=-1)


class EigTrainer(NetTrainer):
    def __init__(self, model, loss_fn, optimizer, scheduler, min_lr):
        super().__init__(model, loss_fn, optimizer, scheduler, min_lr=min_lr)

    def train_batch(self, batch) -> BatchResult:
        x, eig, y = batch
        x, eig, y = x.to(self.device), eig.to(self.device), y.view(-1,).to(self.device)
        self.optimizer.zero_grad()
        y_pred = self.model(x, eig)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            x, eig, y = batch
            x, eig, y = x.to(self.device), eig.to(self.device), y.view(-1, ).to(self.device)
            y_pred = self.model(x, eig)
            loss = self.loss_fn(y_pred, y)
            num_correct = torch.sum(y == torch.argmax(y_pred, dim=-1).view(-1,))
            return BatchResult(loss.item(), num_correct.item())


def train_spectral_net(matrix_size):
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
    ds_test = HybMat40Ds(lbo_files=test_files_lbo, dist_files=test_files_dist, pc_files=test_files_pc, size=matrix_size)

    # ds_train = SpectralWithEig40Ds(train_files, size=matrix_size)
    # ds_test = SpectralWithEig40Ds(test_files, size=matrix_size)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True, num_workers=4)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True, num_workers=4)

    lr = 1e-4
    min_lr = 5e-6
    l2_reg = 0
    our_model = ResNet(BasicBlock, [2, 2, 2, 2])
    # our_model = SpectralSimpleNet(mat_size=matrix_size)
    # our_model = SpectralSimpleNetBn(mat_size=matrix_size)
    # our_model = SpectralFcNet(c_in=matrix_size)
    # our_model = SpectralConv1dNet(c_in=matrix_size)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    trainer = NetTrainer(our_model, loss_fn, optimizer, scheduler, min_lr=min_lr)
    # trainer = EigTrainer(our_model, loss_fn, optimizer, scheduler, min_lr=min_lr)

    expr_name = f'Spectral-resnet-brs-lr{lr}-noise'
    if os.path.isfile(f'results/{expr_name}.pt'):
        os.remove(f'results/{expr_name}.pt')
    _ = trainer.fit(dl_train, dl_test, num_epochs=10000, early_stopping=50, checkpoints=expr_name)
    return


if __name__ == '__main__':
    train_spectral_net(matrix_size=64)
