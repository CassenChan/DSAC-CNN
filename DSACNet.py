import torch.nn as nn
import torch
from torchsummary import summary
from thop import profile
from torchstat import stat
from ptflops import get_model_complexity_info

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        return out*x

class ChannelAttention1(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        return out*x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out*x


class DSACNet(nn.Module):   #卷积计算图层大小时，除不尽则向下取整
    def __init__(self, num_classes=7):
        super(DSACNet, self).__init__()
        self.conv1_1_dw = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3)
        )
        self.conv1_1_pw = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU())
        self.conv1_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, groups=3, bias=False),
            nn.BatchNorm2d(3),
        )
        self.conv1_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU())
        self.pooling_1 = nn.Sequential(nn.AdaptiveMaxPool2d((64, 64)))


        self.conv2_1_dw = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1, groups=12, bias=False),
            nn.BatchNorm2d(12)
        )
        self.conv2_1_pw = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.conv2_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=0, groups=12, bias=False),
            nn.BatchNorm2d(12)
        )
        self.conv2_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.pooling_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32, 32)))


        self.conv3_1_dw = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32)
        )
        self.conv3_1_pw = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv3_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, groups=32, bias=False),
            nn.BatchNorm2d(32)
        )
        self.conv3_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.pooling_3 = nn.Sequential(nn.AdaptiveMaxPool2d((16, 16)))


        self.conv4_1_dw = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64)
        )
        self.conv4_1_pw = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv4_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, groups=64, bias=False),
            nn.BatchNorm2d(64)
        )
        self.conv4_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.pooling_4 = nn.Sequential(nn.AdaptiveMaxPool2d((8, 8)))


        self.conv5_1_dw = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv5_1_pw = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv5_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, groups=256, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv5_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.pooling_5 = nn.Sequential(nn.AdaptiveMaxPool2d((4, 4)))

        self.sa = SpatialAttention()
        self.ca1 = ChannelAttention1(6)
        self.ca2 = ChannelAttention(16)
        self.ca3 = ChannelAttention(32)
        self.ca4 = ChannelAttention(128)
        self.ca5 = ChannelAttention(256)

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        x1_1 = self.conv1_1_dw(x)
        x1_1 = self.sa(x1_1)
        x1_1 = self.conv1_1_pw(x1_1)
        x1_1 = self.ca1(x1_1)
        x1_2 = self.conv1_2_dw(x)
        x1_2 = self.sa(x1_2)
        x1_2 = self.conv1_2_pw(x1_2)
        x1_2 = self.ca1(x1_2)
        x = torch.cat([x1_1, x1_2], dim=1)
        x = self.pooling_1(x)

        x2_1 = self.conv2_1_dw(x)
        x2_1 = self.sa(x2_1)
        x2_1 = self.conv2_1_pw(x2_1)
        x2_1 = self.ca2(x2_1)
        x2_2 = self.conv2_2_dw(x)
        x2_2 = self.sa(x2_2)
        x2_2 = self.conv2_2_pw(x2_2)
        x2_2 = self.ca2(x2_2)
        x = torch.cat([x2_1, x2_2], dim=1)
        x = self.pooling_2(x)

        x3_1 = self.conv3_1_dw(x)
        x3_1 = self.sa(x3_1)
        x3_1 = self.conv3_1_pw(x3_1)
        x3_1 = self.ca3(x3_1)
        x3_2 = self.conv3_2_dw(x)
        x3_2 = self.sa(x3_2)
        x3_2 = self.conv3_2_pw(x3_2)
        x3_2 = self.ca3(x3_2)
        x = torch.cat([x3_1, x3_2], dim=1)
        x = self.pooling_3(x)

        x4_1 = self.conv4_1_dw(x)
        x4_1 = self.sa(x4_1)
        x4_1 = self.conv4_1_pw(x4_1)
        x4_1 = self.ca4(x4_1)
        x4_2 = self.conv4_2_dw(x)
        x4_2 = self.sa(x4_2)
        x4_2 = self.conv4_2_pw(x4_2)
        x4_2 = self.ca4(x4_2)
        x = torch.cat([x4_1, x4_2], dim=1)
        x = self.pooling_4(x)

        x5_1 = self.conv5_1_dw(x)
        x5_1 = self.sa(x5_1)
        x5_1 = self.conv5_1_pw(x5_1)
        x5_1 = self.ca5(x5_1)
        x5_2 = self.conv5_2_dw(x)
        x5_2 = self.sa(x5_2)
        x5_2 = self.conv5_2_pw(x5_2)
        x5_2 = self.ca5(x5_2)
        x = torch.cat([x5_1, x5_2], dim=1)
        x = self.pooling_5(x)

        x = self.classifier(x)
        x = x.view(x.size()[0], -1)

        return x

if __name__ == '__main__':
    net = DSACNet()

    # stat(net, (3, 128, 128))
    # net = net.cuda()
    # summary(net, (3, 128, 128))
    input = torch.randn(1, 3, 128, 128)
    # output = net(input)
    # print("The net out: ", output)
    flops, params = profile(net, inputs=(input, ))
    print(flops, params)


    # #计算方式1
    # flops, params = profile(net, inputs=(input, ))
    # print(flops, params)

    # #计算方式2
    # flops, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True, print_per_layer_stat=True)
    # print("|flops: %s |params: %s" % (flops, params))

    # #计算方式3
    # stat(net, (3, 128, 128))

    # #计算方式4
    # net = net.cuda()
    # summary(net, (3, 128, 128))
