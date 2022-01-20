import torch
import torch.nn as nn


class SharedFCBlock(nn.Module):
    def __init__(self, in_features, out_features, normalize, activation):
        super(SharedFCBlock, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, 1, bias=not normalize)
        self.bn = nn.BatchNorm1d(out_features) if normalize else None
        self.af = activation

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y = self.conv(y)
        if self.bn:
            y = self.bn(y)
        if self.af:
            y = self.af(y)
        y = y.permute(0, 2, 1)
        return y


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, normalize, activation):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=not normalize)
        self.bn = nn.BatchNorm1d(out_features) if normalize else None
        self.af = activation

    def forward(self, x):
        y = self.fc(x)
        if self.bn:
            y = self.bn(y)
        if self.af:
            y = self.af(y)
        return y


class TNet(nn.Module):
    def __init__(self, lnet, gnet, matrix_size):
        super(TNet, self).__init__()
        self.matrix_size = matrix_size
        self.lnet = nn.Sequential(*[SharedFCBlock(**b) for b in lnet])
        self.gnet = nn.Sequential(*[FCBlock(**b) for b in gnet])
        self.fc = nn.Linear(gnet[-1]['out_features'], matrix_size ** 2)

    def forward(self, x):
        batch_size = x.size(0)
        y = self.lnet(x)
        y = torch.max(y, 1, True)[0].view(batch_size, -1)
        y = self.gnet(y)
        y = self.fc(y)
        y = y.view(-1, self.matrix_size, self.matrix_size)
        return y


class Transform(nn.Module):
    def __init__(self, lnet, gnet, matrix_size):
        super(Transform, self).__init__()
        self.tnet = TNet(lnet, gnet, matrix_size)

    def forward(self, x):
        w = self.tnet(x)
        y = torch.bmm(x, w)
        return y, w


class Extraction(nn.Module):
    def __init__(self, blocks):
        super(Extraction, self).__init__()
        self.net = nn.Sequential(*[SharedFCBlock(**b) for b in blocks])

    def forward(self, x):
        return self.net(x)


class Globalization(nn.Module):
    def __init__(self):
        super(Globalization, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return torch.max(x, 1, True)[0].view(batch_size, -1)


class Classification(nn.Module):
    def __init__(self, blocks):
        super(Classification, self).__init__()
        self.net = nn.Sequential(*[FCBlock(**b) for b in blocks])

    def forward(self, x):
        return self.net(x)


class PointNet(nn.Module):
    def __init__(self, net):
        super(PointNet, self).__init__()
        blocks = []
        for block in net:
            if block['type'] == 'transform':
                blocks.append(Transform(matrix_size=block['matrix_size'], **block['blocks']))
            elif block['type'] == 'extraction':
                blocks.append(Extraction(block['blocks']))
            elif block['type'] == 'globalization':
                blocks.append(Globalization())
            elif block['type'] == 'classification':
                blocks.append(Classification(block['blocks']))
        self.net = nn.ModuleList(blocks)

    def forward(self, x):
        ws = []
        for i, block in enumerate(self.net):
            outputs = block(x if i == 0 else y)
            if isinstance(block, Transform):
                y = outputs[0]
                ws.append(outputs[1])
            else:
                y = outputs
        return y, ws
