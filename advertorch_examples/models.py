import math
import torch.nn as nn
import torch.nn.functional as F


class LeNet5Madry(nn.Module):
    # model replicated from
    #   https://github.com/MadryLab/mnist_challenge/blob/
    #   2527d24c4c34e511a12b8a9d7cf6b949aae6fc1b/model.py
    # TODO: combine with the model in advertorch.test_utils

    def __init__(
            self, nb_filters=(1, 32, 64), kernel_sizes=(5, 5),
            paddings=(2, 2), strides=(1, 1), pool_sizes=(2, 2),
            nb_hiddens=(7 * 7 * 64, 1024), nb_classes=10):
        super(LeNet5Madry, self).__init__()
        self.conv1 = nn.Conv2d(
            nb_filters[0], nb_filters[1], kernel_size=kernel_sizes[0],
            padding=paddings[0], stride=strides[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(pool_sizes[0])
        self.conv2 = nn.Conv2d(
            nb_filters[1], nb_filters[2], kernel_size=kernel_sizes[1],
            padding=paddings[0], stride=strides[0])
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(pool_sizes[1])
        self.linear1 = nn.Linear(nb_hiddens[0], nb_hiddens[1])
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(nb_hiddens[1], nb_classes)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


def get_lenet5madry_with_width(widen_factor):
    return LeNet5Madry(
        nb_filters=(1, int(widen_factor * 32), int(widen_factor * 64)),
        nb_hiddens=(7 * 7 * int(widen_factor * 64), int(widen_factor * 1024)))


# WideResNet related code adapted from
#   https://github.com/xternalz/WideResNet-pytorch/blob/
#   ae12d25bdf273010bd4a54971948a6c796cb95ed/wideresnet.py


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.drop_rate = drop_rate
        self.in_out_equal = (in_planes == out_planes)

        if not self.in_out_equal:
            self.conv_shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.in_out_equal:
            x = self.conv_shortcut(out)
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out += x
        return out


class ConvGroup(nn.Module):
    def __init__(
            self, num_blocks, in_planes, out_planes, block, stride,
            drop_rate=0.0):
        super(ConvGroup, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, num_blocks, stride, drop_rate)

    def _make_layer(
            self, block, in_planes, out_planes, num_blocks, stride, drop_rate):
        layers = []
        for i in range(int(num_blocks)):
            layers.append(
                block(in_planes=in_planes if i == 0 else out_planes,
                      out_planes=out_planes,
                      stride=stride if i == 0 else 1,
                      drop_rate=drop_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0,
                 color_channels=3, block=BasicBlock):
        super(WideResNet, self).__init__()
        num_channels = [
            16, int(16 * widen_factor),
            int(32 * widen_factor), int(64 * widen_factor)]
        assert((depth - 4) % 6 == 0)
        num_blocks = (depth - 4) / 6

        self.conv1 = nn.Conv2d(
            color_channels, num_channels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.convgroup1 = ConvGroup(
            num_blocks, num_channels[0], num_channels[1], block, 1, drop_rate)
        self.convgroup2 = ConvGroup(
            num_blocks, num_channels[1], num_channels[2], block, 2, drop_rate)
        self.convgroup3 = ConvGroup(
            num_blocks, num_channels[2], num_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                mod.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                mod.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.convgroup1(out)
        out = self.convgroup2(out)
        out = self.convgroup3(out)
        out = self.relu(self.bn1(out))
        out = out.mean(dim=-1).mean(dim=-1)
        out = self.fc(out)
        return out


def get_cifar10_wrn28_widen_factor(widen_factor):
    from advertorch.utils import PerImageStandardize
    model = WideResNet(28, 10, widen_factor)
    model = nn.Sequential(PerImageStandardize(), model)
    return model
