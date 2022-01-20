import torch
import torch.nn as nn


class OrthogonalRegularizer(nn.Module):
    def __init__(self, device):
        super(OrthogonalRegularizer, self).__init__()
        self.device = device

    def forward(self, w):
        batch_size, nrows, _ = w.size()
        e = torch.eye(nrows, device=self.device).repeat((batch_size,  1, 1))
        m = e - torch.bmm(w, w.permute(0, 2, 1))
        return torch.sum(m * m)
