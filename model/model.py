import torch.nn as nn
import torch
import torch.nn.functional as F


def weight_zero(layer):
    nn.init.zeros_(layer.weight)


def weight_xavier(layer):
    nn.init.xavier_uniform_(layer.weight)


def bias_zero(layer):
    nn.init.zeros_(layer.bias)


class View1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: (B, 1024, 1)
        :return: (B, 1024)
        """
        return x.view(x.shape[0], x.shape[-2])


class View2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: (B, 20, 4, 4, M)
        :return: (B, 320, M)
        """
        return x.view(x.shape[0], -1, x.shape[-1])


class AddEye3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: (B, 3 * 3)
        :return: (B, 3, 3)
        """
        eye = torch.eye(3, device=x.device).view(1, 3, 3)
        return x.view(x.shape[0], 3, 3).add(eye)


class AddEye64(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: (B, 64 * 64)
        :return: (B, 64, 64)
        """
        eye = torch.eye(64, device=x.device).view(1, 64, 64)
        return x.view(x.shape[0], 64, 64).add(eye)


class PointNetBase(nn.Module):
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
        filters = (3, 64, 64, 64, 128, 1024)
        affine = (filters[-1], 512, 256, 40)
        for in_filters, c_out in zip(filters, filters[1:]):
            modules.extend([nn.Conv1d(in_filters, c_out, kernel_size=1),
                            nn.ReLU(),
                            nn.BatchNorm1d(c_out)])
        modules.append(nn.MaxPool1d(kernel_size=1024))
        modules.append(View1D())
        for in_affine, out_affine in zip(affine[:-1], affine[1:-1]):
            modules.extend([nn.Linear(in_affine, out_affine),
                           nn.ReLU(),
                           nn.BatchNorm1d(out_affine)])
        modules.append(nn.Dropout(p=0.7))
        modules.append(nn.Linear(affine[-2], affine[-1]))
        modules.append(nn.LogSoftmax(dim=-1))
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class Trans3(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N) represents batch of N (usually N=1024) (x, y, z).
        After Conv1d filters: (B, 1024, N).
        After pooling: (B, 1024, 1).
        After View: (B, 1024).
        Linear layers: (B, 1024) -> (B, 3, 3).
        """
        super().__init__()
        # Transform 3x3
        trans3_layers = []
        filters = (3, 64, 128, 1024)
        for c_in, c_out in zip(filters, filters[1:]):
            trans3_layers.extend([nn.Conv1d(c_in, c_out, kernel_size=1).apply(bias_zero).apply(weight_xavier),
                                  nn.BatchNorm1d(c_out),
                                  nn.ReLU()])
        trans3_layers.extend([nn.MaxPool1d(kernel_size=1024), View1D()])
        affine = (filters[-1], 512, 256)
        for in_affine, out_affine in zip(affine, affine[1:]):
            trans3_layers.extend([nn.Linear(in_affine, out_affine).apply(bias_zero).apply(weight_xavier),
                                  nn.BatchNorm1d(out_affine),
                                  nn.ReLU()])
        trans3_layers.extend([nn.Linear(affine[-1], 9).apply(weight_zero).apply(bias_zero),
                              AddEye3()])
        self.trans3_seq = nn.Sequential(*trans3_layers)

    def forward(self, x):
        return self.trans3_seq(x)


class Trans64(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 64, N).
        After Conv1d filters: (B, 1024, N).
        After pooling: (B, 1024, 1).
        After View: (B, 1024).
        Linear layers: (B, 1024) -> (B, 64, 64).
        """
        super().__init__()
        # Transform 64x64
        trans64_layers = []
        filters = (64, 64, 128, 1024)
        for c_in, c_out in zip(filters, filters[1:]):
            trans64_layers.extend([nn.Conv1d(c_in, c_out, kernel_size=1).apply(bias_zero).apply(weight_xavier),
                                  nn.BatchNorm1d(c_out),
                                  nn.ReLU()])
        trans64_layers.extend([nn.MaxPool1d(kernel_size=1024), View1D()])
        affine = (filters[-1], 512, 256)
        for in_affine, out_affine in zip(affine, affine[1:]):
            trans64_layers.extend([nn.Linear(in_affine, out_affine).apply(bias_zero).apply(weight_xavier),
                                  nn.BatchNorm1d(out_affine),
                                  nn.ReLU()])
        trans64_layers.extend([nn.Linear(affine[-1], 64 ** 2).apply(weight_zero).apply(bias_zero),
                               AddEye64()])
        self.trans64_seq = nn.Sequential(*trans64_layers)

    def forward(self, x):
        return self.trans64_seq(x)


class PointNetFeatures(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N) represents batch of N (usually N=1024) (x, y, z).
        After Conv1d filters: (B, 1024, N).
        After pooling: (B, 1024, 1).
        After View: (B, 1024).
        """
        super().__init__()
        self.trans3 = Trans3()
        self.trans64 = Trans64()

        # MLP 64 sequence
        mlp64_layers = []
        filters = (3, 64, 64)
        for c_in, c_out in zip(filters, filters[1:]):
            mlp64_layers.extend([nn.Conv1d(c_in, c_out, kernel_size=1).apply(bias_zero).apply(weight_xavier),
                                 nn.BatchNorm1d(c_out),
                                 nn.ReLU()])
        self.mlp64_seq = nn.Sequential(*mlp64_layers)

        feature_layers = []
        filters = (64, 128, 1024)
        for c_in, c_out in zip(filters, filters[1:]):
            feature_layers.extend([nn.Conv1d(c_in, c_out, kernel_size=1).apply(bias_zero).apply(weight_xavier),
                                   nn.BatchNorm1d(c_out),
                                   nn.ReLU()])
        feature_layers.extend([nn.MaxPool1d(kernel_size=1024),
                               View1D()])
        self.get_features = nn.Sequential(*feature_layers)

    def forward(self, x):
        trans3 = self.trans3(x)  # (B, 3, N) --> (B, 3, 3)
        x = trans3.bmm(x)  # (B, 3, 3) @ (B, 3, N)
        x = self.mlp64_seq(x)
        trans64 = self.trans64(x)
        x = trans64.bmm(x)
        return self.get_features(x), trans64


class PointNet(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N) represents batch of N (usually N=1024) (x, y, z).
        After Conv1d filters: (B, 1024, N).
        After pooling: (B, 1024, 1).
        After View: (B, 1024).
        Linear layers: (B, 1024) -> (B, num_classes). num_classes = 40 for ModelNet40 data-set.
        """
        super().__init__()
        self.feature_net = PointNetFeatures()

        affine_layers = []
        channels = (1024, 512, 256)
        for c_in, c_out in zip(channels, channels[1:]):
            affine_layers.extend([nn.Linear(c_in, c_out).apply(bias_zero).apply(weight_xavier),
                                  nn.BatchNorm1d(c_out),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3)])
        affine_layers.append(nn.Linear(channels[-1], 40).apply(bias_zero).apply(weight_xavier))
        self.affine_seq = nn.Sequential(*affine_layers)

    def forward(self, x):
        x, trans64 = self.feature_net(x)
        return self.affine_seq(x), trans64


class PicNetFeatures(nn.Module):
    def __init__(self, m=6, s=32):
        """
        Input dim: (B, 1, S, S, M)
        Output features dim: (B, 20 * (S/4-3) ** 2)
        """
        super().__init__()
        self.m = m
        self.s = s
        modules = []  # (B, 1, S, S, M)
        modules.extend([nn.Conv3d(1, 10, kernel_size=(5, 5, 1)),  # (B, 10, S-4, S-4, M)
                        nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 10, S/2-2, S/2-2, M)
                        nn.ReLU()])
        modules.extend([nn.Conv3d(10, 20, kernel_size=(5, 5, 1)),  # (B, 20, S/2-6, S/2-6, M)
                        nn.Dropout3d(),
                        nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 20, S/4-3, S/4-3, M)
                        nn.ReLU()])
        modules.extend([View2D(),  # (B, 20 * (S/4-3) ** 2, M)
                        nn.MaxPool1d(kernel_size=self.m),  # (B, 20 * (S/4-3) ** 2, 1)
                        View1D()])  # (B, 20 * (S/4-3) ** 2)
        self.feature_seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.feature_seq(x)

    def feature_len(self):
        return int(20 * (self.s/4 - 3) ** 2)


class PicNet(nn.Module):
    def __init__(self, m=6, s=40):
        super().__init__()
        self.feature_net = PicNetFeatures(m=m, s=s)
        c_in = self.feature_net.feature_len()
        modules = []
        modules.extend([nn.Linear(c_in, 256),  # (B, 256)
                        nn.ReLU(),
                        nn.Dropout()])
        modules.extend([nn.Linear(256, 128),  # (B, 256)
                        nn.ReLU(),
                        nn.Dropout()])
        modules.append(nn.Linear(128, 40))  # (B, 40)
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(self.feature_net(x))


class CuppNet(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N)
        Output dim: (B, num_classes)
        """
        super().__init__()
        self.proj_feature_net = PicNetFeatures()  # (B, 500)
        proj_feature_len = self.proj_feature_net.feature_len()
        self.pc_feature_net = PointNetFeatures()  # (B, 1024)

        proj_layers = [nn.Linear(proj_feature_len, 256),
                       nn.ReLU(),
                       nn.Dropout()]
        self.proj_seq = nn.Sequential(*proj_layers)  # (B, 256)

        pc_channels = (1024, 512, 256)
        pc_layers = []
        for c_in, c_out in zip(pc_channels, pc_channels[1:]):
            pc_layers.extend([nn.Linear(c_in, c_out),
                              nn.ReLU(),
                              nn.BatchNorm1d(c_out)])
        self.pc_seq = nn.Sequential(*pc_layers)  # (B, 256)

        affine = (512, 256, 40)
        affine_layers = []
        for c_in, c_out in zip(affine[:-1], affine[1:-1]):
            affine_layers.extend([nn.Linear(c_in, c_out),
                                  nn.ReLU(),
                                  nn.Dropout()])
        affine_layers.append(nn.Linear(affine[-2], affine[-1]))
        self.affine_seq = nn.Sequential(*affine_layers)

    def forward(self, pc, proj):
        pc, trans64 = self.pc_feature_net(pc)
        pc_feat = self.pc_seq(pc)
        proj_feat = self.proj_seq(self.proj_feature_net(proj))
        return self.affine_seq(torch.cat([pc_feat, proj_feat], dim=-1)), trans64


class CuppNetMax(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N)
        Output dim: (B, num_classes)
        """
        super().__init__()
        self.proj_feature_net = PicNetFeatures()
        proj_feature_len = self.proj_feature_net.feature_len()
        self.pc_feature_net = PointNetFeatures()

        proj_layers = [nn.Linear(proj_feature_len, 256),
                       nn.ReLU(),
                       nn.BatchNorm1d(256)]
        self.proj_seq = nn.Sequential(*proj_layers)  # (B, 512)

        pc_layers = [nn.Linear(1024, 256),
                     nn.ReLU(),
                     nn.BatchNorm1d(256)]
        self.pc_seq = nn.Sequential(*pc_layers)  # (B, 512)

        affine = (256, 128)
        affine_layers = []
        affine_layers.extend([nn.MaxPool1d(kernel_size=2),  # (B, 512, 2) --> (B, 512, 1)
                              View1D()])  # (B, 512, 1) --> (B, 512)
        for c_in, c_out in zip(affine, affine[1:]):
            affine_layers.extend([nn.Linear(c_in, c_out),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(c_out),
                                  nn.Dropout()])
        affine_layers.append(nn.Linear(128, 40))
        self.affine_seq = nn.Sequential(*affine_layers)

    def forward(self, pc, proj):
        pc, trans64 = self.pc_feature_net(pc)
        pc_feat = self.pc_seq(pc)
        proj_feat = self.proj_seq(self.proj_feature_net(proj))
        return self.affine_seq(torch.stack([pc_feat, proj_feat], dim=-1)), trans64


class CuppNetSumProb(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N)
        Output dim: (B, num_classes)
        """
        super().__init__()
        self.point_net = PointNet()
        # self.pic_net = PicNet()
        self.pic_net = PicResNetVox()

    def forward(self, pc, proj):
        pc, trans64 = self.point_net(pc)
        proj = self.pic_net(proj)
        prob = (F.log_softmax(pc, dim=-1) + F.log_softmax(proj, dim=-1)) / 2
        return prob, trans64


class PicResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input_size (B, 1, S, S, M)
        self.resnet_features = ResNet(BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Linear(512, 40)

    def forward(self, x):
        x0 = self.resnet_features(x[:, :, :, :, 0])
        x1 = self.resnet_features(x[:, :, :, :, 1])
        x2 = self.resnet_features(x[:, :, :, :, 2])
        x3 = self.resnet_features(x[:, :, :, :, 3])
        x4 = self.resnet_features(x[:, :, :, :, 4])
        x5 = self.resnet_features(x[:, :, :, :, 5])
        x = torch.stack((x0, x1, x2, x3, x4, x5), dim=-1)
        x = F.max_pool1d(x, kernel_size=6)
        x = x.view(x.shape[0], x.shape[1])
        return self.fc(x)


class PicResNetView(nn.Module):
    """
    input_size (B, num_slices, S, S)
    """
    def __init__(self):
        super().__init__()
        # self.resnet_features = ResNet(BasicBlock, [2, 2, 2, 2])  # (B, 1, S, S)
        self.resnet_features = ResNet(BasicBlock, [2, 2, 2, 2])  # (B, 1, S, S)
        # nn.Conv1d(num_slices, 1, kernel_size=1),
        # nn.ReLU(),
        # nn.BatchNorm1d(c_out)
        # num_features = 512
        # affine_layers = []
        # affine_layers.extend([nn.MaxPool1d(kernel_size=2),  # (B, 512, 2) --> (B, 512, 1)
        #                       View1D()])  # (B, 512, 1) --> (B, 512)
        # for c_in, c_out in zip(affine, affine[1:]):
        # affine_layers.extend([nn.Linear(num_features * num_slices, num_features),
        #                       nn.ReLU(),
        #                       nn.BatchNorm1d(c_out),
        #                       nn.Dropout()])
        # affine_layers.append(nn.Linear(128, 40))
        # self.fc = nn.Linear(num_features * num_slices, num_features)

    def forward(self, x):
        num_slices = x.shape[1]
        x = torch.stack([self.resnet_features(x[:, i:i+1, :, :]) for i in range(num_slices)], dim=-1)
        x = F.max_pool1d(x, kernel_size=num_slices)
        x = x.view(x.shape[0], x.shape[1])
        # x = x.view(x.shape[0], -1)
        # return F.relu(self.fc(x))
        return x


class PicResNetVox(nn.Module):
    """
    input_size (B, num_slices, S, S, num_views)
    """
    def __init__(self):
        super().__init__()
        self.resnet_view = PicResNetView()
        self.fc = nn.Linear(512, 40)
        # affine = (512*3, 512, 256)
        # affine_layers = []
        # for c_in, c_out in zip(affine, affine[1:]):
        #     affine_layers.extend([nn.Linear(c_in, c_out),
        #                           nn.BatchNorm1d(c_out),
        #                           nn.ReLU(),
        #
        #                           nn.Dropout()])
        # affine_layers.append(nn.Linear(256, 40))
        # self.affine_seq = nn.Sequential(*affine_layers)

    def forward(self, x):
        num_views = x.shape[-1]
        x = torch.stack([self.resnet_view(x[:, :, :, :, i]) for i in range(num_views)], dim=-1)
        x = F.max_pool1d(x, kernel_size=num_views)
        x = x.view(x.shape[0], x.shape[1])
        return self.fc(x)


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
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
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
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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
        # x = self.fc(x)
        return x

