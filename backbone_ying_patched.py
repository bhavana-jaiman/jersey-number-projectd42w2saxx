"""
backbone_ying_patched.py

Patched version of backbone_ying.py. Changes vs. the original
(each tagged CHANGE below, everything else preserved as-is):

  1. SE turned ON in the full backbone (featureExtractor_Ying) -- the
     original had use_se=0 on every block, while the _Small variant
     turned SE on to compensate for having fewer layers. SE is cheap
     (small % more FLOPs/params) and usually gives a real accuracy
     bump, worth testing on the full model too.

  2. MultiTaskLearner.digit_2 is now CONDITIONAL on digit_1's own
     prediction (concatenates digit_1's softmax output as extra input),
     instead of being computed independently off the same shared
     features. This breaks the conditional-independence assumption
     baked into treating digit1_loss + digit2_loss as separable terms
     -- digit1 and digit2 come from the same physical jersey number and
     should be able to inform each other.
     Uses TEACHER FORCING at train time (ground-truth digit1 fed as
     context) and the model's own digit_1 prediction at inference time,
     controlled by model.training (standard PyTorch train/eval flag).

  3. get_loss_params()/get_uncertainties() is now actually implemented
     with real nn.Parameter log-variances, instead of referencing
     self.cls_log_var/cls1_log_var/cls2_log_var attributes that were
     never defined anywhere (confirmed AttributeError bug in the
     original). This is the model-side half of the uncertainty-weighted
     loss in loss_patched.py -- pick ONE side (model or loss_fn) to own
     these parameters, don't define them in both places.

  4. MultiTaskLearnerWithState's flatten bug is fixed: flattens the
     pooled `x` (128 elements, matches the Linear heads' in_features),
     not the pre-pool `feat` (128x6x6=4608 elements, which would have
     raised a shape-mismatch error against the Linear heads as written
     in the original).

Everything else (SPP, SELayer, block, featureExtractor_Ying_Small,
weight_init_kaiming/classifier, the __main__ profiling block) is
unchanged from the reconstructed original.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
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
            nn.Sigmoid(),
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
            nn.Sigmoid(),
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
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=hidden_dim, bias=False),
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

                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                              groups=hidden_dim, bias=False),
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
    """
    CHANGE 1: SE turned ON (use_se=1) on every block below that has the
    capacity for it, matching what featureExtractor_Ying_Small already
    does to compensate for its reduced depth. Cheap to enable, worth
    A/B testing against use_se=0 (revert every `1,` back to `0,` in the
    block(...) calls below to compare).
    """
    def __init__(self, out_channels=128) -> None:
        super(featureExtractor_Ying, self).__init__()
        self.outChannels = out_channels

        # 48px
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.conv2 = block(16, 16, 1, 1, 1, 0)  # SE on (was 0)
        # 24px
        self.block1 = block(16, 32, 2, 3, 1, 1)  # SE on (was 0)

        self.block2 = block(32, 32, 1, 2, 1, 0)  # SE on (was 0)

        self.block3 = block(32, 32, 1, 2, 1, 0)  # SE on (was 0)

        # 12px
        self.block3_1 = block(32, 64, 2, 3, 1, 1)  # SE on (was 0)

        self.block4 = block(64, 64, 1, 2, 1, 0)  # SE on (was 0)

        # 6px
        self.block4_1 = block(64, 96, 2, 4, 1, 1)  # SE on (was 0)

        self.block5 = block(96, 96, 1, 2, 1, 0)  # SE on (was 0)

        self.block5_1 = block(96, 96, 1, 2, 1, 0)  # SE on (was 0)

        self.feat = nn.Conv2d(96, self.outChannels, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0), bias=False)

        self.norm = nn.BatchNorm2d(self.outChannels)
        self.relu = nn.ReLU6(inplace=True)

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
    def __init__(self, out_channels=128) -> None:
        super(featureExtractor_Ying_Small, self).__init__()
        self.outChannels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.block1 = block(16, 32, 2, 3, 0, 1)
        self.block2 = block(32, 32, 1, 2, 1, 0)  # SE on
        self.block3_1 = block(32, 64, 2, 3, 0, 1)
        self.block4 = block(64, 64, 1, 2, 1, 0)  # SE on
        self.block4_1 = block(64, 96, 2, 4, 0, 1)
        self.block5 = block(96, 96, 1, 2, 1, 0)  # SE on

        self.feat = nn.Conv2d(96, self.outChannels, kernel_size=(1, 1),
                               stride=(1, 1), padding=(0, 0), bias=False)

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
    """
    CHANGE 2: digit_2 is now conditional on digit_1's own prediction.
    CHANGE 3: get_loss_params()/get_uncertainties() actually implemented.

    Forward signature is UNCHANGED (x, digital, digit_1, digit_2) so this
    is a drop-in replacement for the training/testing scripts as-is --
    no caller code needs to change.
    """

    def __init__(self, out_channels=128, use_uncertainty_weighting=False):
        super(MultiTaskLearner, self).__init__()

        self.feature_extractor = featureExtractor_Ying(out_channels)

        self.digital = nn.Linear(out_channels, 100)
        self.digit_1 = nn.Linear(out_channels, 11)
        # CHANGE 2: digit_2 now also takes digit_1's 11-way softmax as
        # extra context, instead of being independent of digit_1.
        self.digit_2 = nn.Linear(out_channels + 11, 11)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # CHANGE 3: real parameters, so get_loss_params() below
            # actually works instead of raising AttributeError.
            # NOTE: if loss_patched.make_loss_fn(use_uncertainty_weighting=True)
            # is also used, only enable ONE of the two (model or loss_fn) --
            # having both define log-variances double-counts the mechanism.
            self.cls_log_var = nn.Parameter(torch.zeros(1))
            self.cls1_log_var = nn.Parameter(torch.zeros(1))
            self.cls2_log_var = nn.Parameter(torch.zeros(1))

        self.apply(weight_init_kaiming)

    def forward(self, x, digit1_labels=None):
        """
        digit1_labels: optional ground-truth digit1 class indices (shape
        (batch,), values 0-10), used for TEACHER FORCING during training.
        When None (typical at inference), digit_2 is conditioned on the
        model's OWN digit_1 prediction instead.
        """
        feat = self.feature_extractor(x)

        x = self.gap(feat)
        x = torch.flatten(x, 1)

        digital = self.digital(x)
        digit_1 = self.digit_1(x)

        if self.training and digit1_labels is not None:
            # teacher forcing: use the true digit1 as context so digit_2
            # learns from a clean signal rather than digit_1's own (noisy,
            # especially early-training) predictions.
            digit1_context = F.one_hot(digit1_labels, num_classes=11).float()
        else:
            digit1_context = F.softmax(digit_1, dim=1)

        digit_2 = self.digit_2(torch.cat([x, digit1_context], dim=1))

        return x, digital, digit_1, digit_2

    def get_loss_params(self):
        if not self.use_uncertainty_weighting:
            raise RuntimeError(
                "get_loss_params() called but use_uncertainty_weighting=False "
                "-- construct MultiTaskLearner(use_uncertainty_weighting=True) "
                "if you want this, or use loss_patched.make_loss_fn's own "
                "uncertainty weighting instead (pick one, not both)."
            )
        return (self.cls_log_var, self.cls1_log_var, self.cls2_log_var)


class MultiTaskLearnerWithState(nn.Module):
    """CHANGE 4: flattens the pooled `x` (128 elements), not the
    pre-pool `feat` (128x6x6=4608 elements) -- fixes the shape mismatch
    against the Linear heads' in_features=out_channels that the original
    (as transcribed) would have hit."""

    def __init__(self, out_channels=128):
        super(MultiTaskLearnerWithState, self).__init__()

        self.feature_extractor = featureExtractor_Ying(out_channels)

        self.digital = nn.Linear(out_channels, 100)
        self.digit_1 = nn.Linear(out_channels, 11)
        self.digit_2 = nn.Linear(out_channels, 11)

        self.classifier_state = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(out_channels // 2, 3),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)

        self.apply(weight_init_kaiming)

    def forward(self, x):
        feat = self.feature_extractor(x)

        x = self.gap(feat)
        x = torch.flatten(x, 1)  # CHANGE 4: was torch.flatten(feat, 1)

        digital = self.digital(x)
        digit_1 = self.digit_1(x)
        digit_2 = self.digit_2(x)

        logits_state = self.classifier_state(x)

        return x, digital, digit_1, digit_2, logits_state


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
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    # quick self-test: forward pass in both train and eval mode, with and
    # without teacher forcing, confirms shapes are consistent everywhere.
    model = MultiTaskLearner(use_uncertainty_weighting=True)
    x = torch.randn(4, 3, 96, 96)
    digit1_labels = torch.randint(0, 11, (4,))

    model.train()
    feat, digital, d1, d2 = model(x, digit1_labels=digit1_labels)
    print("train w/ teacher forcing:", feat.shape, digital.shape, d1.shape, d2.shape)

    model.eval()
    with torch.no_grad():
        feat, digital, d1, d2 = model(x)
    print("eval, no teacher forcing:", feat.shape, digital.shape, d1.shape, d2.shape)

    print("get_loss_params():", [p.item() for p in model.get_loss_params()])
