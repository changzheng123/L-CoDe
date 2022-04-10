import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d


def conv1x1(in_planes, out_planes, bias=True):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1_relu(in_planes, out_planes, bias=True):
    """1x1 convolution with padding and relu"""
    block = nn.Sequential(
        conv1x1(in_planes, out_planes, bias),
        nn.ReLU(inplace=True)
    )
    return block


def conv3x3_bn_relu(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
    return block


def conv3x3_bn(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.BatchNorm2d(out_planes),
    )
    return block


def conv3x3_tanh(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding and tanh"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.Tanh()
    )
    return block

def conv3x3_bn_relu_maxpool(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    return block

def conv3x3_bn_relu_avgpool(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
    return block

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block