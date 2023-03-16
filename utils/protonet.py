import torch
import torch.nn as nn
from utils.resnet import *

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 use_bn=True,
                 max_pool=None,
                 activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_max_pool = (max_pool is not None)
        self.use_activation = (activation is not None)

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)
        if self.use_max_pool:
            self.max_pool = torch.nn.MaxPool2d(max_pool)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        if self.use_max_pool:
            x = self.max_pool(x)
        return x


def get_activation(activation):
    activation = activation.lower()
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "relu6":
        return torch.nn.ReLU6(inplace=True)
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()

    else:
        raise NotImplementedError("Activation {} not implemented".format(activation))


class ProtoNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, input_dim, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.block1 = ConvBlock(input_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block2 = ConvBlock(hid_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block3 = ConvBlock(hid_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block4 = ConvBlock(hid_dim, z_dim, 3, max_pool=2, padding=1)

    def forward(self, x):
        sample_size = x.size(1)
        input = torch.chunk(x,sample_size,dim=1)
        for i, value in enumerate(input):
            tmp = self.block1(value.squeeze(1))
            tmp = self.block2(tmp)
            tmp = self.block3(tmp)
            tmp = self.block4(tmp)
            tmp = tmp.view(tmp.size(0), -1)
            if i == 0:
                output = tmp.unsqueeze(1)
            else :
                output = torch.concat([output,tmp.unsqueeze(1)],dim=1)
        return output

class ProtoNet_withResNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, num_classes=1024):
        super(ProtoNet_withResNet, self).__init__()
        self.net = ResNet34(num_classes=num_classes)

    def forward(self, x):
        sample_size = x.size(1)
        input = torch.chunk(x,sample_size,dim=1)
        for i, value in enumerate(input):
            tmp = self.net(value.squeeze(1))
            tmp = tmp.view(tmp.size(0), -1)
            if i == 0:
                output = tmp.unsqueeze(1)
            else :
                output = torch.concat([output,tmp.unsqueeze(1)],dim=1)
        return output