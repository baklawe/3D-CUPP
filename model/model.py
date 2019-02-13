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
        self.pic_net = PicNet()

    def forward(self, pc, proj):
        pc, trans64 = self.point_net(pc)
        proj = self.pic_net(proj)
        prob = (F.log_softmax(pc, dim=-1) + F.log_softmax(proj, dim=-1)) / 2
        return prob, trans64
