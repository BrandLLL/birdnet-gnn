"""
Export the trained U-Net to a TorchScript .pt consumable by vol2bird's
libmistnet (libmistnet/libmistnet.cpp -> run_mistnet).

The C code (lib/librender.c) does, after run_mistnet returns a flat array:
    create4DTensor(out, N_CLASSES, n_elev, 608, 608)   # [class][elev][x][y]
and indexes tensor[NONBIO_INDEX] / tensor[BIO_INDEX].

So the exported module must output a tensor whose flattened order is
class-major: all (elev,x,y) for class 0, then all for class 1, i.e. shape
    (1, N_CLASSES, n_elev, 608, 608)  ->  flattens to 2*5*608*608.

The internal UNet produces (1, n_elev*2, 608, 608) ordered elevation-major,
so this wrapper reshapes and permutes, then applies sigmoid to emit
probabilities (the C side thresholds probabilities directly).

Usage:
    python export_torchscript.py --ckpt unet_radar_weights.pt --out unet_radar.pt
"""

import argparse
import torch
import torch.nn as nn

from unet import UNet, N_CLASSES, N_ELEV, IN_CHANNELS, DIMENSION


class ExportWrapper(nn.Module):
    """Wraps UNet to emit class-major probability tensor for the C interface."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        # x: (1, 15, 608, 608)
        logits = self.net(x)                       # (1, N_ELEV*2, H, W), elev-major
        n, _, h, w = logits.shape
        # (1, N_ELEV, N_CLASSES, H, W)
        logits = logits.view(n, N_ELEV, N_CLASSES, h, w)
        # -> class-major (1, N_CLASSES, N_ELEV, H, W)
        logits = logits.permute(0, 2, 1, 3, 4).contiguous()
        probs = torch.sigmoid(logits)
        return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="trained UNet state_dict")
    p.add_argument("--out", default="unet_radar.pt", help="output TorchScript file")
    args = p.parse_args()

    net = UNet()
    state = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    wrapper = ExportWrapper(net).eval()
    example = torch.randn(1, IN_CHANNELS, DIMENSION, DIMENSION)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    traced.save(args.out)

    # sanity check
    with torch.no_grad():
        out = traced(example)
    assert tuple(out.shape) == (1, N_CLASSES, N_ELEV, DIMENSION, DIMENSION), out.shape
    print(f"Saved TorchScript model to {args.out}")
    print(f"output shape {tuple(out.shape)} -> flattens to "
          f"{N_CLASSES*N_ELEV*DIMENSION*DIMENSION} floats "
          f"(class-major, matches lib/librender.c).")


if __name__ == "__main__":
    main()
