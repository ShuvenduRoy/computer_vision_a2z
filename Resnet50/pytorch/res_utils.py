import torch
import torch.nn as nn


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleC(nn.Module):
    def __init__(self, nin, nout, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nin != nout
        self.conv = nn.Conv2d(nin, nout, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleD, self).__init__()
    assert stride == 2
    self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
    self.bn   = nn.BatchNorm2d(nOut)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x

