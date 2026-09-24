import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thop import profile
from flopth import flopth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import math


class SPP(nn.Module):
    def __init__(self, out_pool_size) -> None:
        super(SPP, self).__init__()
        self.out_pool_size = out_pool_size

    def forward(self, x):
        out_pool_size = self.out_pool_size
        num_sample, _, h, w = x.size()
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(h / out_pool_size[i]))
            w_wid = int(math.ceil(w / out_pool_size[i]))
            h_pad = math.floor((h_wid * out_pool_size[i] - h + 1) / 2)
            w_pad = math.floor((w_wid * out_pool_size[i] - w + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(
                h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
            x = maxpool(x)
            if (i == 0):
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp


class SELayer(nn.Module):
    def __init__(self, inOutChannel, reduction=4):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cSE = nn.Sequential(
            nn.Linear(inOutChannel, inOutChannel // reduction, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(inOutChannel // reduction, inOutChannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.cSE(y).view(b, c, 1, 1)
        return x * y


class SELayer_Conv(nn.Module):
    def __init__(self, inOutChannel, reduction=4):
        super(SELayer_Conv, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cSE = nn.Sequential(
            nn.Conv2d(inOutChannel, inOutChannel // reduction, 1, 1, 0, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inOutChannel // reduction, inOutChannel, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.cSE(self.gap(x))
        return x * y


class block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, use_se, use_exDw, use_ConvNext=False):
        super(block, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_channel * expand_ratio)
        self.identity = stride == 1 and in_channel == out_channel

        self.use_exDw = use_exDw
        self.use_ConvNext = use_ConvNext

        if self.use_exDw:
            self.exDW = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, 1, 1,
                          groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
            )

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(
                    1, 1), padding=(1, 1), groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=1e-05, momentum=0.1,
                               affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True),

                SELayer_Conv(hidden_dim) if use_se else nn.Identity(),

                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            if self.use_ConvNext:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),

                    nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channel),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),

                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride=(
                        1, 1), groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),

                    SELayer_Conv(hidden_dim) if use_se else nn.Identity(),
                    nn.ReLU6(inplace=True),

                    nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channel),
                )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x) if not self.use_exDw else x + self.conv(self.exDW(x))
        else:
            return self.conv(x) if not self.use_exDw else self.conv(self.exDW(x))


class featureExtractor_Ying(nn.Module):
    def __init__(self, out_channels=256) -> None:
        super(featureExtractor_Ying, self).__init__()
        self.outChannels = out_channels

        # 48px
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(
                2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = block(16, 16, 1, 1, 0, 0)  # block1
        # 24px
        self.block1 = block(16, 32, 2, 3, 0, 1)

        self.block2 = block(32, 32, 1, 2, 0, 0)

        self.block3 = block(32, 32, 1, 2, 0, 0)

        # 12px
        self.block3_1 = block(32, 64, 2, 3, 0, 1)

        self.block4 = block(64, 64, 1, 2, 0, 0)

        # 6px
        self.block4_1 = block(64, 96, 2, 4, 0, 1)

        self.block5 = block(96, 96, 1, 2, 0, 0)

        self.block5_1 = block(96, 96, 1, 2, 0, 0)

        self.feat = nn.Conv2d(96, self.outChannels, kernel_size=(
            1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.norm = nn.BatchNorm2d(self.outChannels)
        self.relu = nn.ReLU6(inplace=True)

        # self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block3_1(x)

        x = self.block4(x)

        x = self.block4_1(x)

        x = self.block5(x)

        x = self.block5_1(x)

        x = self.relu(self.norm(self.feat(x)))

        return x


class featureExtractor_Ying_Small(nn.Module):
    def __init__(self, out_channels=256) -> None:
        super(featureExtractor_Ying_Small, self).__init__()
        self.outChannels = out_channels

        # 48px stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(
                2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # 24px
        self.block1 = block(16, 32, 2, 3, 0, 1)  # 2, 3
        self.block2 = block(32, 32, 1, 2, 1, 0)

        # 12px
        self.block3_1 = block(32, 64, 2, 3, 0, 1)  # 2, 3
        self.block4 = block(64, 64, 1, 2, 1, 0)

        # 6px
        self.block4_1 = block(64, 96, 2, 4, 0, 1)  # 2, 4
        self.block5 = block(96, 96, 1, 2, 1, 0)

        self.feat = nn.Conv2d(96, self.outChannels, kernel_size=(
            1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.norm = nn.BatchNorm2d(self.outChannels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3_1(x)

        x = self.block4(x)

        x = self.block4_1(x)

        x = self.block5(x)

        x = self.relu(self.norm(self.feat(x)))

        return x


class MultiTaskLearner(nn.Module):
    def __init__(self, out_channels=256):
        super(MultiTaskLearner, self).__init__()

        self.feature_extractor = featureExtractor_Ying(out_channels)

        # ---------------HCL_Changes---------------
        # Removed whole digit linear head per client request (2025-06-16 architecture review)
        # self.digital = nn.Linear(out_channels, 100)
        self.digit_1 = nn.Linear(out_channels, 10)  # output channels changed from 11 to 10
        self.digit_2 = nn.Linear(out_channels, 11)

        # Added state linear head for state classification per client request
        self.classifier_state = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(out_channels // 2, 3)
        )
        # -------------------------------------------

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)

        self.apply(weight_init_kaiming)

    def forward(self, x):
        feat = self.feature_extractor(x)

        x = self.gap(feat)

        x = torch.flatten(x, 1)

        # ---------------HCL_Changes---------------
        # digital = self.digital(x)
        digit_1 = self.digit_1(x)
        digit_2 = self.digit_2(x)

        logits_state = self.classifier_state(x)
        # -------------------------------------------

        return x, digit_1, digit_2, logits_state  # digital removed from return

    def get_loss_params(self):
        return (self.cls_log_var, self.cls1_log_var, self.cls2_log_var)


class MultiTaskLearnerWithState(nn.Module):
    def __init__(self, out_channels=256):
        super(MultiTaskLearnerWithState, self).__init__()

        self.feature_extractor = featureExtractor_Ying(out_channels)

        # ---------------HCL_Changes---------------
        # Removed whole digit linear head per client request (2025-06-16 architecture review)
        # self.digital = nn.Linear(out_channels, 100)
        self.digit_1 = nn.Linear(out_channels, 10)  # output channels changed from 11 to 10
        self.digit_2 = nn.Linear(out_channels, 11)

        # classifier_state already existed prior to this change request; left as-is
        self.classifier_state = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(out_channels // 2, 3)
        )
        # -------------------------------------------

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)

        self.apply(weight_init_kaiming)

    def forward(self, x):
        feat = self.feature_extractor(x)

        x = self.gap(feat)

        x = torch.flatten(x, 1)

        # ---------------HCL_Changes---------------
        # digital = self.digital(x)
        digit_1 = self.digit_1(x)
        digit_2 = self.digit_2(x)

        logits_state = self.classifier_state(x)
        # -------------------------------------------

        return x, digit_1, digit_2, logits_state  # digital removed from return  #digital


def weight_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    model = MultiTaskLearnerWithState()
    model = model.cuda()

    input = torch.rand(1, 3, 96, 96).cuda()
    macs, _ = profile(model, inputs=(input,))
    print('flops: %.2fG, Params: %.2fM' % (macs * 2 / 1e9, 0))
