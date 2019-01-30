import torch
import torch.nn as nn
import torch.nn.functional as F


class View1D(nn.Module):
    def __init__(self):
        super(View1D, self).__init__()

    def forward(self, x):
        """
        :param x: (B, 1024, 1)
        :return: (B, 1024)
        """
        return x.view(x.shape[0], x.shape[-2])


class PointNet(nn.Module):
    def __init__(self):
        """
        Input dim: (B, 3, N) represents batch of N (usually 1024) (x, y, z).
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
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


