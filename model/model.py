import torch.nn as nn
import torch


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
        :param x: # (B, 20, 4, 4, M)
        :return: # (B, 320, M)
        """
        return x.view(x.shape[0], -1, x.shape[-1])


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
        for in_filters, out_filters in zip(filters, filters[1:]):
            modules.extend([nn.Conv1d(in_filters, out_filters, kernel_size=1),
                            nn.ReLU(),
                            nn.BatchNorm1d(out_filters)])
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


class PicNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = 6
        modules = []  # (B, 1, 28, 28, M)
        modules.extend([nn.Conv3d(1, 10, kernel_size=(5, 5, 1)),  # (B, 10, 24, 24, M)
                        nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 10, 12, 12, M)
                        nn.ReLU()])
        modules.extend([nn.Conv3d(10, 20, kernel_size=(5, 5, 1)),  # (B, 20, 8, 8, M)
                        nn.Dropout3d(),
                        nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 20, 4, 4, M)
                        nn.ReLU()])
        modules.append(View2D())  # (B, 320, M)
        modules.append(nn.MaxPool1d(kernel_size=m))  # (B, 320, 1)
        modules.append(View1D())  # (B, 320)
        modules.extend([nn.Linear(320, 256),  # (B, 50)
                        nn.ReLU(),
                        nn.Dropout()])
        modules.extend([nn.Linear(256, 128),  # (B, 50)
                        nn.ReLU(),
                        nn.Dropout()])
        modules.append(nn.Linear(128, 40))  # (B, 40)
        modules.append(nn.LogSoftmax(dim=-1))
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class PicNetBN(nn.Module):
    def __init__(self):
        super().__init__()
        m = 6
        modules = []  # (B, 1, 28, 28, M)
        modules.extend([nn.Conv3d(1, 10, kernel_size=(5, 5, 1)),  # (B, 10, 24, 24, M)
                        nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 10, 12, 12, M)
                        nn.ReLU()])
        modules.extend([nn.Conv3d(10, 20, kernel_size=(5, 5, 1)),  # (B, 20, 8, 8, M)
                        nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 20, 4, 4, M)
                        nn.ReLU(),
                        nn.BatchNorm3d(20)])
        modules.append(View2D())  # (B, 320, M)
        modules.append(nn.MaxPool1d(kernel_size=m))  # (B, 320, 1)
        modules.append(View1D())  # (B, 320)
        modules.extend([nn.Linear(320, 256),  # (B, 50)
                        nn.ReLU(),
                        nn.BatchNorm1d(256)])
        modules.extend([nn.Linear(256, 128),  # (B, 50)
                        nn.ReLU(),
                        nn.Dropout()])
        modules.append(nn.Linear(128, 40))  # (B, 40)
        modules.append(nn.LogSoftmax(dim=-1))
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class CuppNet(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N) represents batch of N (usually N=1024) (x, y, z).
        After Conv1d filters: (B, 1024, N).
        After pooling: (B, 1024, 1).
        After View: (B, 1024).
        Linear layers: (B, 1024) -> (B, num_classes). num_classes = 40 for ModelNet40 data-set.
        """
        super().__init__()
        pc_layers = []
        pc_filters = (3, 64, 64, 64, 128, 1024)
        pc_affine = (pc_filters[-1], 512, 256)
        for in_filters, out_filters in zip(pc_filters, pc_filters[1:]):
            pc_layers.extend([nn.Conv1d(in_filters, out_filters, kernel_size=1),
                              nn.ReLU(),
                              nn.BatchNorm1d(out_filters)])
        pc_layers.append(nn.MaxPool1d(kernel_size=1024))
        pc_layers.append(View1D())
        for in_affine, out_affine in zip(pc_affine[:-1], pc_affine[1:]):
            pc_layers.extend([nn.Linear(in_affine, out_affine),
                              nn.ReLU(),
                              nn.BatchNorm1d(out_affine)])
        self.pc_seq = nn.Sequential(*pc_layers)  # Outputs (B, 1024) feature vector.

        m = 6
        proj_layers = []  # (B, 1, 28, 28, M)
        proj_layers.extend([nn.Conv3d(1, 10, kernel_size=(5, 5, 1)),  # (B, 10, 24, 24, M)
                            nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 10, 12, 12, M)
                            nn.ReLU()])
        proj_layers.extend([nn.Conv3d(10, 20, kernel_size=(5, 5, 1)),  # (B, 20, 8, 8, M)
                            nn.Dropout3d(),
                            nn.MaxPool3d(kernel_size=(2, 2, 1)),  # (B, 20, 4, 4, M)
                            nn.ReLU()])
        proj_layers.append(View2D())  # (B, 320, M)
        proj_layers.append(nn.MaxPool1d(kernel_size=m))  # (B, 320, 1)
        proj_layers.append(View1D())  # (B, 320)
        proj_layers.extend([nn.Linear(320, 256),
                            nn.ReLU(),
                            nn.Dropout()])
        self.proj_seq = nn.Sequential(*proj_layers)  # Outputs (B, 320) feature vector.

        affine = (256+256, 40)
        affine_layers = []
        for in_affine, out_affine in zip(affine[:-1], affine[1:-1]):
            affine_layers.extend([nn.Linear(in_affine, out_affine),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(out_affine)])
        affine_layers.append(nn.Dropout(p=0.7))
        affine_layers.append(nn.Linear(affine[-2], affine[-1]))
        affine_layers.append(nn.LogSoftmax(dim=-1))
        self.affine_seq = nn.Sequential(*affine_layers)

    def forward(self, pc, proj):
        pc_feat = self.pc_seq(pc)
        proj_feat = self.proj_seq(proj)
        return self.affine_seq(torch.cat([pc_feat, proj_feat], dim=-1))
