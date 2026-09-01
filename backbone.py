import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Spatial Pyramid Pooling
# ============================================================

class SPP(nn.Module):
    def __init__(self, out_pool_size=(1, 2, 4)):
        super().__init__()
        self.out_pool_size = out_pool_size

    def forward(self, x):
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)

        spp = []

        for pool_size in self.out_pool_size:
            h_kernel = int(math.ceil(h / pool_size))
            w_kernel = int(math.ceil(w / pool_size))

            h_pad = int(
                math.floor(
                    (h_kernel * pool_size - h + 1) / 2
                )
            )

            w_pad = int(
                math.floor(
                    (w_kernel * pool_size - w + 1) / 2
                )
            )

            pooled = F.max_pool2d(
                x,
                kernel_size=(h_kernel, w_kernel),
                stride=(h_kernel, w_kernel),
                padding=(h_pad, w_pad)
            )

            spp.append(pooled.view(batch_size, -1))

        return torch.cat(spp, dim=1)


# ============================================================
# SE Attention
# ============================================================

class SELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


# ============================================================
# ECA Attention
# ============================================================

class ECALayer(nn.Module):
    """
    Efficient Channel Attention.

    Much lighter than SE because it avoids the
    dimensionality-reduction FC layers.
    """

    def __init__(self, channels, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        # B,C,1,1 -> B,1,C
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)

        # B,1,C -> B,C,1,1
        y = y.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


# ============================================================
# Selectable Attention
# ============================================================

class SelectiveAttention(nn.Module):

    def __init__(
        self,
        channels,
        attention="none",
        reduction=4,
        eca_kernel=3
    ):
        super().__init__()

        attention = attention.lower()

        if attention == "se":
            self.attention = SELayer(
                channels,
                reduction=reduction
            )

        elif attention == "eca":
            self.attention = ECALayer(
                channels,
                k_size=eca_kernel
            )

        elif attention == "both":
            self.attention = nn.Sequential(
                SELayer(channels, reduction),
                ECALayer(channels, eca_kernel)
            )

        else:
            self.attention = nn.Identity()

    def forward(self, x):
        return self.attention(x)


# ============================================================
# Lightweight Spatial Transformer Network
# ============================================================

class LightweightSTN(nn.Module):
    """
    Lightweight affine STN.

    Learns:
        rotation
        translation
        scale
        shear

    The final affine layer is initialized to identity,
    therefore the STN initially does almost nothing.
    """

    def __init__(self, input_channels=3):
        super().__init__()

        self.localization = nn.Sequential(

            nn.Conv2d(
                input_channels,
                8,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                24,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        # Identity initialization
        nn.init.zeros_(self.fc_loc[-1].weight)

        self.fc_loc[-1].bias.data.copy_(
            torch.tensor(
                [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0],
                dtype=torch.float
            )
        )

    def forward(self, x):

        xs = self.localization(x)

        theta = self.fc_loc(xs)

        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(
            theta,
            x.size(),
            align_corners=False
        )

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )

        return x


# ============================================================
# Basic MobileNet-style Block
# ============================================================

class Block(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        stride=1,
        expand_ratio=1,
        use_se=False,
        attention="none"
    ):
        super().__init__()

        hidden_dim = int(round(in_channel * expand_ratio))

        self.use_residual = (
            stride == 1 and in_channel == out_channel
        )

        layers = []

        # Expansion
        if expand_ratio != 1:

            layers.extend([
                nn.Conv2d(
                    in_channel,
                    hidden_dim,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Attention inside selected block
        if use_se:
            layers.append(
                SelectiveAttention(
                    hidden_dim,
                    attention=attention
                )
            )

        # Projection
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                out_channel,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv(x)

        if self.use_residual:
            return x + out

        return out


# ============================================================
# Ying Feature Extractor
# ============================================================

class FeatureExtractorYing(nn.Module):

    def __init__(
        self,
        out_channels=128,
        attention="none",
        attention_stages=None
    ):
        super().__init__()

        if attention_stages is None:
            attention_stages = []

        self.attention_stages = set(
            attention_stages
        )

        # ----------------------------------------------------
        # Initial convolution
        # ----------------------------------------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # ----------------------------------------------------
        # Feature blocks
        # ----------------------------------------------------

        self.block1 = Block(
            16, 32,
            stride=2,
            expand_ratio=3
        )

        self.block2 = Block(
            32, 32,
            stride=1,
            expand_ratio=2
        )

        self.block3 = Block(
            32, 32,
            stride=1,
            expand_ratio=2
        )

        self.block3_1 = Block(
            32, 64,
            stride=2,
            expand_ratio=3
        )

        self.block4 = Block(
            64, 64,
            stride=1,
            expand_ratio=2
        )

        self.block4_1 = Block(
            64, 96,
            stride=2,
            expand_ratio=4
        )

        self.block5 = Block(
            96, 96,
            stride=1,
            expand_ratio=2
        )

        self.block5_1 = Block(
            96, 96,
            stride=1,
            expand_ratio=2
        )

        # ----------------------------------------------------
        # Final feature projection
        # ----------------------------------------------------

        self.feat = nn.Conv2d(
            96,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

        # ----------------------------------------------------
        # Selective attention after feature stages
        # ----------------------------------------------------

        self.attention1 = SelectiveAttention(
            64,
            attention=attention
        )

        self.attention2 = SelectiveAttention(
            96,
            attention=attention
        )

        self.attention3 = SelectiveAttention(
            out_channels,
            attention=attention
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block3_1(x)

        if "block3" in self.attention_stages:
            x = self.attention1(x)

        x = self.block4(x)

        x = self.block4_1(x)

        if "block4" in self.attention_stages:
            x = self.attention2(x)

        x = self.block5(x)
        x = self.block5_1(x)

        x = self.feat(x)
        x = self.norm(x)
        x = self.relu(x)

        if "final" in self.attention_stages:
            x = self.attention3(x)

        return x


# ============================================================
# MultiTask Learner (no state head)
# ============================================================

class MultiTaskLearner(nn.Module):

    def __init__(
        self,
        out_channels=128,
        use_stn=False,
        attention="none",
        attention_stages=None,
        num_whole_classes=100,
        num_digit_classes=11,
        dropout=0.0
    ):
        super().__init__()

        self.use_stn = use_stn
        self.attention = attention

        # ----------------------------------------------------
        # Optional STN
        # ----------------------------------------------------

        if use_stn:
            self.stn = LightweightSTN(3)
        else:
            self.stn = nn.Identity()

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        self.feature_extractor = FeatureExtractorYing(
            out_channels=out_channels,
            attention=attention,
            attention_stages=attention_stages
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(
            p=dropout
        )

        # ----------------------------------------------------
        # Heads
        # ----------------------------------------------------

        self.digital = nn.Linear(
            out_channels,
            num_whole_classes
        )

        self.digit_1 = nn.Linear(
            out_channels,
            num_digit_classes
        )

        self.digit_2 = nn.Linear(
            out_channels,
            num_digit_classes
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(
                    m.weight,
                    std=0.001
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Optional STN
        x = self.stn(x)

        # Backbone
        feat = self.feature_extractor(x)

        # Global average pooling
        x = self.gap(feat)

        x = torch.flatten(
            x,
            1
        )

        x = self.dropout(x)

        # Multitask predictions
        whole_logits = self.digital(x)
        digit1_logits = self.digit_1(x)
        digit2_logits = self.digit_2(x)

        return (
            x,
            whole_logits,
            digit1_logits,
            digit2_logits
        )


# ============================================================
# MultiTask Learner + State
# ============================================================

class MultiTaskLearnerWithState(nn.Module):

    def __init__(
        self,
        out_channels=128,
        use_stn=False,
        attention="none",
        attention_stages=None,
        num_whole_classes=100,
        num_digit_classes=11,
        num_state_classes=3,
        dropout=0.0
    ):
        super().__init__()

        self.use_stn = use_stn
        self.attention = attention

        if use_stn:
            self.stn = LightweightSTN(3)
        else:
            self.stn = nn.Identity()

        self.feature_extractor = FeatureExtractorYing(
            out_channels=out_channels,
            attention=attention,
            attention_stages=attention_stages
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.digital = nn.Linear(
            out_channels,
            num_whole_classes
        )

        self.digit_1 = nn.Linear(
            out_channels,
            num_digit_classes
        )

        self.digit_2 = nn.Linear(
            out_channels,
            num_digit_classes
        )

        # State branch
        self.classifier_state = nn.Sequential(
            nn.Linear(
                out_channels,
                out_channels // 2
            ),
            nn.BatchNorm1d(
                out_channels // 2
            ),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(
                out_channels // 2,
                num_state_classes
            )
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(
                    m.weight,
                    std=0.001
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.stn(x)

        feat = self.feature_extractor(x)

        x = self.gap(feat)

        x = torch.flatten(
            x,
            1
        )

        x = self.dropout(x)

        whole_logits = self.digital(x)
        digit1_logits = self.digit_1(x)
        digit2_logits = self.digit_2(x)
        state_logits = self.classifier_state(x)

        return (
            x,
            whole_logits,
            digit1_logits,
            digit2_logits,
            state_logits
        )


# ============================================================
# Model Factory
# ============================================================

def build_model(
    use_state=True,
    use_stn=False,
    attention="none",
    attention_stages=None,
    out_channels=128,
    dropout=0.0,
    num_whole_classes=100,
    num_digit_classes=11,
    num_state_classes=3
):
    """
    Single entry point for constructing the model used by the
    training script. Controlled entirely via the arguments below
    so that architecture experiments never require editing this
    file directly.

    use_state:
        True  -> MultiTaskLearnerWithState (adds classifier_state head)
        False -> MultiTaskLearner (digit1/digit2/whole only)

    use_stn:
        Enables the LightweightSTN before the feature extractor.

    attention:
        "none" | "se" | "eca" | "both"

    attention_stages:
        Subset of ["block3", "block4", "final"] controlling where
        SelectiveAttention is inserted after the backbone stages.
    """

    if attention_stages is None:
        attention_stages = []

    if use_state:

        model = MultiTaskLearnerWithState(
            out_channels=out_channels,
            use_stn=use_stn,
            attention=attention,
            attention_stages=attention_stages,
            num_whole_classes=num_whole_classes,
            num_digit_classes=num_digit_classes,
            num_state_classes=num_state_classes,
            dropout=dropout
        )

    else:

        model = MultiTaskLearner(
            out_channels=out_channels,
            use_stn=use_stn,
            attention=attention,
            attention_stages=attention_stages,
            num_whole_classes=num_whole_classes,
            num_digit_classes=num_digit_classes,
            dropout=dropout
        )

    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        use_state=True,
        use_stn=True,
        attention="eca",
        attention_stages=[
            "block4",
            "final"
        ],
        out_channels=128,
        dropout=0.1
    )

    model = model.to(device)

    x = torch.randn(
        2,
        3,
        96,
        96
    ).to(device)

    output = model(x)

    print("\nModel test successful")

    for i, out in enumerate(output):
        print(
            f"Output {i}: "
            f"{tuple(out.shape)}"
        )
