"""
backbone_ying_rep.py

RepVGG-based version of the Ying jersey-number backbone.

IMPORTANT DESIGN CHOICE
-----------------------
This file preserves the ORIGINAL Ying backbone's:
    - input size: 3 x 96 x 96
    - channel schedule
    - downsampling schedule
    - final 128-channel feature output
    - compatibility with the existing MultiTaskLearner

The INTERNAL BLOCK TYPE is changed to RepVGG-style blocks.

Training form of RepVGGBlock:
    3x3 Conv + BN
          +
    1x1 Conv + BN
          +
    Identity + BN (when possible)
          |
         ADD
          |
        ReLU6

Deployment form:
    one 3x3 Conv + bias
          |
        ReLU6

This lets us train with a multi-branch block and later fuse it into a
single 3x3 convolution.

The backbone can also run controlled experiments where only selected
blocks are RepVGG blocks and the remaining blocks use the original Ying
`block` implementation.

Examples
--------
1) Full RepVGG backbone:
    model = featureExtractor_Ying_Rep(rep_locations="all")

2) Only deepest block:
    model = featureExtractor_Ying_Rep(rep_locations="block5_1")

3) Last two blocks:
    model = featureExtractor_Ying_Rep(
        rep_locations=["block5", "block5_1"]
    )

4) Last N RepVGG blocks:
    model = featureExtractor_Ying_Rep(num_rep_blocks=2)

5) No RepVGG blocks (useful as a control):
    model = featureExtractor_Ying_Rep(rep_locations="none")

For the BASE RepVGG experiment, use:
    rep_locations="all"

The output remains:
    [B, 128, 6, 6]

so the existing MultiTaskLearner can continue doing:
    GAP -> flatten -> Linear(128, 100/11/11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the original Ying block so that mixed/ablation experiments can
# preserve the existing implementation exactly.
try:
    from .backbone_ying import block as YingBlock
except ImportError:
    try:
        from subModules.backbone_ying import block as YingBlock
    except ImportError:
        try:
            from backbone_ying import block as YingBlock
        except ImportError:
            YingBlock = None


# ============================================================
# 1. RepVGG STRUCTURAL RE-PARAMETERIZATION BLOCK
# ============================================================

class RepVGGBlock(nn.Module):
    """
    RepVGG-style block.

    TRAIN:
        x
        ├── 3x3 Conv + BN ──┐
        ├── 1x1 Conv + BN ──┼── Add -> ReLU6
        └── Identity + BN ──┘

    DEPLOY:
        x -> one 3x3 Conv + bias -> ReLU6

    Identity branch exists only when:
        stride == 1 and in_channels == out_channels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        deploy=False,
    ):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError(
                f"RepVGGBlock supports stride 1 or 2, got {stride}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy

        self.nonlinearity = nn.ReLU6(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            )
        else:
            # 3x3 branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # 1x1 branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # Identity branch is possible only for shape-preserving blocks.
            if stride == 1 and in_channels == out_channels:
                self.rbr_identity = nn.BatchNorm2d(out_channels)
            else:
                self.rbr_identity = None

    # --------------------------------------------------------
    # Conv + BN fusion
    # --------------------------------------------------------

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        """
        Fuse:
            Conv -> BN

        into:
            Conv with bias

        using BN running statistics.
        """
        kernel = conv.weight

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)

        scale = (gamma / std).reshape(-1, 1, 1, 1)

        fused_kernel = kernel * scale
        fused_bias = beta - running_mean * gamma / std

        return fused_kernel, fused_bias

    # --------------------------------------------------------
    # 1x1 -> 3x3
    # --------------------------------------------------------

    @staticmethod
    def _pad_1x1_to_3x3(kernel):
        """
        Put a 1x1 kernel in the center of a 3x3 kernel.
        """
        if kernel is None:
            return None

        if kernel.size(2) == 3:
            return kernel

        return F.pad(
            kernel,
            [1, 1, 1, 1]
        )

    # --------------------------------------------------------
    # Identity -> 3x3
    # --------------------------------------------------------

    def _get_identity_kernel(self):
        """
        Construct the convolution kernel equivalent of Identity.

        Shape:
            [C, C, 3, 3]

        Center element is 1 for matching input/output channels.
        """
        input_dim = self.in_channels

        kernel = torch.zeros(
            (
                input_dim,
                input_dim,
                3,
                3,
            ),
            dtype=self.rbr_identity.weight.dtype,
            device=self.rbr_identity.weight.device,
        )

        for i in range(input_dim):
            kernel[i, i, 1, 1] = 1.0

        return kernel

    def _fuse_identity_bn(self, bn):
        """
        Fuse:
            Identity -> BN

        into:
            3x3 Conv + bias
        """
        kernel = self._get_identity_kernel()

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)

        scale = (gamma / std).reshape(-1, 1, 1, 1)

        fused_kernel = kernel * scale
        fused_bias = beta - running_mean * gamma / std

        return fused_kernel, fused_bias

    # --------------------------------------------------------
    # Equivalent kernel/bias
    # --------------------------------------------------------

    def get_equivalent_kernel_bias(self):
        """
        Calculate the single 3x3 convolution equivalent to all
        training branches.
        """
        if self.deploy:
            return (
                self.rbr_reparam.weight,
                self.rbr_reparam.bias,
            )

        # 3x3 branch
        kernel_3x3, bias_3x3 = self._fuse_conv_bn(
            self.rbr_dense[0],
            self.rbr_dense[1],
        )

        # 1x1 branch
        kernel_1x1, bias_1x1 = self._fuse_conv_bn(
            self.rbr_1x1[0],
            self.rbr_1x1[1],
        )

        kernel_1x1 = self._pad_1x1_to_3x3(kernel_1x1)

        # Identity branch
        if self.rbr_identity is not None:
            kernel_identity, bias_identity = self._fuse_identity_bn(
                self.rbr_identity
            )
        else:
            kernel_identity = torch.zeros_like(kernel_3x3)
            bias_identity = torch.zeros_like(bias_3x3)

        # Structural re-parameterization
        equivalent_kernel = (
            kernel_3x3
            + kernel_1x1
            + kernel_identity
        )

        equivalent_bias = (
            bias_3x3
            + bias_1x1
            + bias_identity
        )

        return equivalent_kernel, equivalent_bias

    # --------------------------------------------------------
    # Convert training block -> deployment block
    # --------------------------------------------------------

    @torch.no_grad()
    def switch_to_deploy(self):
        """
        Convert the training-time multi-branch block into one
        3x3 convolution.

        ALWAYS call model.eval() before conversion.
        """
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )

        self.rbr_reparam.weight.copy_(kernel)
        self.rbr_reparam.bias.copy_(bias)

        del self.rbr_dense
        del self.rbr_1x1

        if self.rbr_identity is not None:
            del self.rbr_identity

        self.deploy = True

    def forward(self, x):
        if self.deploy:
            out = self.rbr_reparam(x)
        else:
            out = self.rbr_dense(x)
            out = out + self.rbr_1x1(x)

            if self.rbr_identity is not None:
                out = out + self.rbr_identity(x)

        return self.nonlinearity(out)


# ============================================================
# 2. REP LOCATION PARSER
# ============================================================

REP_BLOCK_NAMES = (
    "conv2",
    "block1",
    "block2",
    "block3",
    "block3_1",
    "block4",
    "block4_1",
    "block5",
    "block5_1",
)


def _parse_rep_locations(rep_locations):
    """
    Convert user input to a set of block names.

    Accepted:
        "all"
        "none"
        "block5_1"
        "block5,block5_1"
        ["block5", "block5_1"]
        None
    """
    if rep_locations is None:
        return set()

    if isinstance(rep_locations, str):
        value = rep_locations.strip().lower()

        if value in ("", "none", "false", "0"):
            return set()

        if value == "all":
            return set(REP_BLOCK_NAMES)

        locations = {
            x.strip()
            for x in value.split(",")
            if x.strip()
        }

    else:
        locations = {
            str(x).strip()
            for x in rep_locations
            if str(x).strip()
        }

    invalid = locations.difference(REP_BLOCK_NAMES)

    if invalid:
        raise ValueError(
            f"Unknown RepVGG block(s): {sorted(invalid)}\n"
            f"Valid names: {REP_BLOCK_NAMES}"
        )

    return locations


def _locations_from_num(num_rep_blocks):
    """
    Select the deepest N blocks.

    Example:
        num_rep_blocks=1
            -> {"block5_1"}

        num_rep_blocks=2
            -> {"block5", "block5_1"}

        num_rep_blocks=4
            -> {"block4", "block4_1", "block5", "block5_1"}
    """
    if num_rep_blocks is None:
        return None

    num_rep_blocks = int(num_rep_blocks)

    if num_rep_blocks < 0 or num_rep_blocks > len(REP_BLOCK_NAMES):
        raise ValueError(
            f"num_rep_blocks must be between 0 and "
            f"{len(REP_BLOCK_NAMES)}"
        )

    if num_rep_blocks == 0:
        return set()

    return set(REP_BLOCK_NAMES[-num_rep_blocks:])


# ============================================================
# 3. YING + REPVGG BACKBONE
# ============================================================

class featureExtractor_Ying_Rep(nn.Module):
    """
    RepVGG-based Ying backbone.

    Channel/spatial schedule preserved from the supplied Ying backbone:

        Input       : 3 x 96 x 96

        conv1       : 3  -> 16, stride 2
                      16 x 48 x 48

        conv2       : 16 -> 16, stride 1
                      16 x 48 x 48

        block1      : 16 -> 32, stride 2
                      32 x 24 x 24

        block2      : 32 -> 32, stride 1
                      32 x 24 x 24

        block3      : 32 -> 32, stride 1
                      32 x 24 x 24

        block3_1    : 32 -> 64, stride 2
                      64 x 12 x 12

        block4      : 64 -> 64, stride 1
                      64 x 12 x 12

        block4_1    : 64 -> 96, stride 2
                      96 x 6 x 6

        block5      : 96 -> 96, stride 1
                      96 x 6 x 6

        block5_1    : 96 -> 96, stride 1
                      96 x 6 x 6

        feat        : 96 -> 128, 1x1
                      128 x 6 x 6

    Existing MultiTaskLearner compatibility:
        GAP(128 x 6 x 6) -> 128
        Linear(128, 100)
        Linear(128, 11)
        Linear(128, 11)

    Parameters
    ----------
    out_channels:
        Final feature dimension. Keep 128 to use the existing heads.

    rep_locations:
        Exact locations that should use RepVGG blocks.

        Examples:
            "all"
            "block5_1"
            "block5,block5_1"
            ["block4", "block4_1", "block5", "block5_1"]
            "none"

    num_rep_blocks:
        Alternative to rep_locations.

        Selects the deepest N blocks.

            1 -> block5_1
            2 -> block5 + block5_1
            4 -> block4 + block4_1 + block5 + block5_1

        If num_rep_blocks is supplied, it takes precedence over
        rep_locations.

    deploy:
        False:
            training-time RepVGG multi-branch blocks.

        True:
            expects the checkpoint to already contain fused
            3x3 deployment convolutions.

    use_original_for_non_rep:
        If True, blocks not selected for RepVGG use the original Ying
        `block` implementation.

        This is useful for controlled experiments such as:
            rep_locations="block5_1"

        In that case only block5_1 becomes RepVGG and all other blocks
        remain exactly as in the original Ying backbone.

        DEFAULT = True.

        For the BASE RepVGG experiment, simply use:
            rep_locations="all"

        Then every backbone block is RepVGG and this flag has no effect.
    """

    def __init__(
        self,
        out_channels=128,
        rep_locations="all",
        num_rep_blocks=None,
        deploy=False,
        use_original_for_non_rep=True,
    ):
        super().__init__()

        self.outChannels = out_channels
        self.deploy = deploy

        # --------------------------------------------------------
        # Decide which blocks are RepVGG blocks
        # --------------------------------------------------------

        if num_rep_blocks is not None:
            self.rep_locations = _locations_from_num(
                num_rep_blocks
            )
        else:
            self.rep_locations = _parse_rep_locations(
                rep_locations
            )

        self.use_original_for_non_rep = (
            use_original_for_non_rep
        )

        if (
            self.use_original_for_non_rep
            and YingBlock is None
            and len(self.rep_locations) != len(REP_BLOCK_NAMES)
        ):
            raise ImportError(
                "Could not import the original Ying `block` class. "
                "Use the file as part of the subModules package or "
                "set use_original_for_non_rep=False."
            )

        # --------------------------------------------------------
        # Stem
        # Same 3 -> 16, stride 2 as supplied Ying backbone.
        # --------------------------------------------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # --------------------------------------------------------
        # Blocks
        # --------------------------------------------------------

        self.conv2 = self._make_block(
            "conv2",
            16,
            16,
            stride=1,
            expand_ratio=1,
            use_se=0,
            use_exDw=0,
        )

        self.block1 = self._make_block(
            "block1",
            16,
            32,
            stride=2,
            expand_ratio=3,
            use_se=0,
            use_exDw=1,
        )

        self.block2 = self._make_block(
            "block2",
            32,
            32,
            stride=1,
            expand_ratio=2,
            use_se=0,
            use_exDw=0,
        )

        self.block3 = self._make_block(
            "block3",
            32,
            32,
            stride=1,
            expand_ratio=2,
            use_se=0,
            use_exDw=0,
        )

        self.block3_1 = self._make_block(
            "block3_1",
            32,
            64,
            stride=2,
            expand_ratio=2,
            use_se=0,
            use_exDw=1,
        )

        self.block4 = self._make_block(
            "block4",
            64,
            64,
            stride=1,
            expand_ratio=2,
            use_se=0,
            use_exDw=0,
        )

        self.block4_1 = self._make_block(
            "block4_1",
            64,
            96,
            stride=2,
            expand_ratio=4,
            use_se=0,
            use_exDw=1,
        )

        self.block5 = self._make_block(
            "block5",
            96,
            96,
            stride=1,
            expand_ratio=2,
            use_se=0,
            use_exDw=0,
        )

        self.block5_1 = self._make_block(
            "block5_1",
            96,
            96,
            stride=1,
            expand_ratio=2,
            use_se=0,
            use_exDw=0,
        )

        # --------------------------------------------------------
        # Final 1x1 projection
        # Same 96 -> 128 as supplied Ying backbone.
        # --------------------------------------------------------

        self.feat = nn.Conv2d(
            96,
            self.outChannels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(
            self.outChannels
        )

        self.relu = nn.ReLU6(
            inplace=True
        )

        self.apply(self._init_weights)

    # ------------------------------------------------------------
    # Block factory
    # ------------------------------------------------------------

    def _make_block(
        self,
        name,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
        use_se,
        use_exDw,
    ):
        if name in self.rep_locations:
            # BASE RepVGG PATH
            return RepVGGBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                deploy=self.deploy,
            )

        # Optional control experiment:
        # keep the original Ying block at locations that are not
        # selected for RepVGG.
        if self.use_original_for_non_rep:
            return YingBlock(
                in_channels,
                out_channels,
                stride,
                expand_ratio,
                use_se,
                use_exDw,
            )

        # If a block is not selected and the user requested the
        # original Ying implementation, preserve that block exactly.
        if self.use_original_for_non_rep:
            if YingBlock is None:
                raise ImportError(
                    "Could not import the original Ying `block` class. "
                    "This is required when use_original_for_non_rep=True "
                    "and not all blocks are RepVGG."
                )

            return YingBlock(
                in_channels,
                out_channels,
                stride,
                expand_ratio,
                use_se,
                use_exDw,
            )

        # Optional mode: every non-selected block is also RepVGG.
        return RepVGGBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            deploy=self.deploy,
        )

    # ------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if m.bias is not None:
                nn.init.constant_(
                    m.bias,
                    0.0,
                )

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(
                m.weight,
                1.0,
            )
            nn.init.constant_(
                m.bias,
                0.0,
            )

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

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

        x = self.feat(x)
        x = self.norm(x)
        x = self.relu(x)

        return x

    # ------------------------------------------------------------
    # Information helpers
    # ------------------------------------------------------------

    def get_feature_dim(self):
        return self.outChannels

    def get_rep_locations(self):
        return sorted(
            self.rep_locations,
            key=lambda x: REP_BLOCK_NAMES.index(x)
        )

    # ------------------------------------------------------------
    # Deploy conversion
    # ------------------------------------------------------------

    @torch.no_grad()
    def switch_to_deploy(self):
        """
        Convert every RepVGGBlock in this backbone to one 3x3 Conv.

        Call:
            model.eval()
            model.switch_to_deploy()
        """
        self.eval()

        converted = []

        for name, module in self.named_modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()
                converted.append(name)

        self.deploy = True

        return converted


# ============================================================
# 4. ALIASES / FACTORY
# ============================================================

FeatureExtractor_Ying_Rep = featureExtractor_Ying_Rep
RepVGGBackbone = featureExtractor_Ying_Rep


def build_backbone_ying_rep(
    out_channels=128,
    rep_locations="all",
    num_rep_blocks=None,
    deploy=False,
    use_original_for_non_rep=True,
):
    """
    Factory function for the RepVGG Ying backbone.
    """
    return featureExtractor_Ying_Rep(
        out_channels=out_channels,
        rep_locations=rep_locations,
        num_rep_blocks=num_rep_blocks,
        deploy=deploy,
        use_original_for_non_rep=use_original_for_non_rep,
    )


# ============================================================
# 5. TESTS
# ============================================================

def check_output_shape():
    """
    Check that the new backbone has the same output shape expected
    by the current MultiTaskLearner.

    Expected:
        input  = [2, 3, 96, 96]
        output = [2, 128, 6, 6]
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = featureExtractor_Ying_Rep(
        out_channels=128,
        rep_locations="all",
        deploy=False,
    ).to(device)

    model.eval()

    x = torch.randn(
        2,
        3,
        96,
        96,
        device=device,
    )

    with torch.no_grad():
        y = model(x)

    print("=" * 70)
    print("REP-VGG YING BACKBONE SHAPE TEST")
    print("=" * 70)
    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("Expected output: (2, 128, 6, 6)")

    assert y.shape == (
        2,
        128,
        6,
        6,
    ), (
        f"Unexpected output shape: {tuple(y.shape)}"
    )

    print("PASS: output shape is correct.")

    return model, x, y


def check_train_deploy_equivalence():
    """
    Verify that the multi-branch training model and the fused
    deployment model produce nearly identical outputs.

    IMPORTANT:
        model.eval() is used before conversion because deployment
        fusion uses BatchNorm running statistics.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = featureExtractor_Ying_Rep(
        out_channels=128,
        rep_locations="all",
        deploy=False,
    ).to(device)

    model.eval()

    x = torch.randn(
        2,
        3,
        96,
        96,
        device=device,
    )

    with torch.no_grad():
        y_before = model(x)

    model.switch_to_deploy()

    with torch.no_grad():
        y_after = model(x)

    max_diff = (
        y_before - y_after
    ).abs().max().item()

    mean_diff = (
        y_before - y_after
    ).abs().mean().item()

    print("\n" + "=" * 70)
    print("TRAIN -> DEPLOY EQUIVALENCE TEST")
    print("=" * 70)
    print(
        f"Max absolute difference : {max_diff:.10e}"
    )
    print(
        f"Mean absolute difference: {mean_diff:.10e}"
    )

    # This is intentionally a practical numerical tolerance.
    # If this fails, do NOT train a large model yet; inspect fusion.
    assert max_diff < 1e-4, (
        "Train/deploy conversion difference is too large: "
        f"{max_diff}"
    )

    print("PASS: structural re-parameterization is numerically correct.")

    return max_diff, mean_diff


def print_architecture_summary():
    """
    Print RepVGG locations and output dimensions.
    """
    model = featureExtractor_Ying_Rep(
        out_channels=128,
        rep_locations="all",
        deploy=False,
    )

    print("\n" + "=" * 70)
    print("REP-VGG YING ARCHITECTURE")
    print("=" * 70)

    print(
        "RepVGG blocks:",
        model.get_rep_locations()
    )

    print(
        "Final feature dimension:",
        model.get_feature_dim()
    )

    print(
        "Input:",
        "(B, 3, 96, 96)"
    )

    print(
        "Backbone output:",
        "(B, 128, 6, 6)"
    )

    print(
        "Existing MultiTaskLearner:",
        "GAP -> 128 -> [100, 11, 11]"
    )


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    print_architecture_summary()

    model, x, y = check_output_shape()

    check_train_deploy_equivalence()

    print("\nAll backbone tests completed successfully.")
