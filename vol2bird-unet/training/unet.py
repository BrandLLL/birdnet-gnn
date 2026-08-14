"""
U-Net segmentation model for vol2bird / MistNet replacement.

Input : 15 channels = 3 radar variables (DBZ, VRAD, WRAD) x 5 elevation angles
        spatial size 608 x 608 (MISTNET_DIMENSION), see lib/constants.h
Output: 2 channels per elevation = {non-biological, biological}

The C side (lib/librender.c) expects the flattened output ordered as
    [class][elev][x][y]  ->  2 * 5 * 608 * 608
so the network produces a tensor of shape (N, n_elev * 2, 608, 608) which,
for a single radar volume (N=1), flattens to exactly that layout when the
channel axis is ordered as (class-major, elev-minor). See export_torchscript.py
for the wrapper that enforces this ordering.

This file defines the architecture only. Channel ordering for the C interface
is handled in the exported wrapper.
"""

import torch
import torch.nn as nn

# Must match lib/constants.h
N_RADAR_VARS = 3          # DBZ, VRAD, WRAD
N_ELEV = 5                # MISTNET_N_ELEV
IN_CHANNELS = N_RADAR_VARS * N_ELEV          # 15
N_CLASSES = 2                                # non-biological, biological
OUT_CHANNELS = N_CLASSES * N_ELEV            # 10 (one 2-class map per elevation)
DIMENSION = 608           # MISTNET_DIMENSION


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # in_ch is the number of channels coming up from the deeper layer
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 608 is divisible by 2 four times (608,304,152,76,38) so sizes align,
        # but pad defensively in case of odd input sizes.
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard 4-level U-Net.

    Output shape: (N, OUT_CHANNELS, H, W) = (N, 10, 608, 608).
    Channel axis is ordered elevation-major here: [e0c0, e0c1, e1c0, e1c1, ...].
    The TorchScript export wrapper re-orders to class-major for the C interface.
    Logits are returned (no softmax/sigmoid); the loss applies activations.
    """

    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8)
        self.up2 = Up(base * 8, base * 4)
        self.up3 = Up(base * 4, base * 2)
        self.up4 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    net = UNet()
    dummy = torch.randn(1, IN_CHANNELS, DIMENSION, DIMENSION)
    out = net(dummy)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"input  : {tuple(dummy.shape)}")
    print(f"output : {tuple(out.shape)}  (expected (1, {OUT_CHANNELS}, {DIMENSION}, {DIMENSION}))")
    print(f"params : {n_params/1e6:.1f}M")
