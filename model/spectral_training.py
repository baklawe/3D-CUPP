import os
from typing import List
import numpy as np
import h5py
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from experiments import get_files_list
from training import NetTrainer


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


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


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
        # print(f'x.shape={x.shape}')
        x = x.view(bs, self.num_features)
        return self.seq(x)


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


def train_spectral_net(matrix_size):
    bs_train, bs_test = 32, 32
    train_files = get_files_list('../data/spectral/spectral_train_files.txt')
    test_files = get_files_list('../data/spectral/spectral_test_files.txt')

    ds_train = Spectral40Ds(train_files, size=matrix_size)
    ds_test = Spectral40Ds(test_files, size=matrix_size)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=True)

    lr = 0.5*1e-3
    min_lr = 1e-5
    l2_reg = 0
    # our_model = ResNet(BasicBlock, [2, 2, 2, 2])
    # our_model = SpectralSimpleNet(mat_size=matrix_size)
    our_model = SpectralSimpleNetBn(mat_size=matrix_size)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(our_model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    trainer = NetTrainer(our_model, loss_fn, optimizer, scheduler, min_lr=min_lr)

    expr_name = f'Spectral-resnet-'
    if os.path.isfile(f'results/{expr_name}.pt'):
        os.remove(f'results/{expr_name}.pt')
    fit_res = trainer.fit(dl_train, dl_test, num_epochs=10000, early_stopping=50, checkpoints=expr_name)
    return


if __name__ == '__main__':
    train_spectral_net(matrix_size=32)
