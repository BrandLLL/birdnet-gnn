"""
Training script for the U-Net biological/non-biological radar segmenter.

This is a runnable skeleton. You must supply a Dataset that yields:
    x      : FloatTensor (15, 608, 608)  -- DBZ/VRAD/WRAD x 5 elevations
    y      : FloatTensor (10, 608, 608)  -- one-hot, elevation-major
                                            [e_i_nonbio, e_i_bio] for i in 0..4
    mask   : FloatTensor (1, 608, 608)   -- optional, 1 where labelled

Channel ordering for x must match lib/librender.c::polarVolumeTo3DTensor,
which writes channel index  iElev + N_ELEV * iVar  with
    iVar 0 = DBZ, 1 = VRAD, 2 = WRAD.
So x = [dbz_e0..dbz_e4, vrad_e0..vrad_e4, wrad_e0..wrad_e4].

Usage:
    python train.py --data /path/to/data --epochs 50 --out unet_radar.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader, Dataset

from unet import UNet, IN_CHANNELS, OUT_CHANNELS, DIMENSION
from loss import BCEDiceLoss


class RadarSegDataset(Dataset):
    """Placeholder. Replace __getitem__ with your real data loading."""

    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        # TODO: index your samples here
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: load and normalize one radar volume + its label
        # x: (15,608,608), y: (10,608,608) one-hot, mask: (1,608,608)
        raise NotImplementedError("Provide your radar volume + label loading.")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet().to(device)

    # pos_weight optionally counters class imbalance (biology is usually rarer)
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.full((OUT_CHANNELS,), args.pos_weight, device=device)
    criterion = BCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=pos_weight,
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_ds = RadarSegDataset(args.data, "train")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)

    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        for batch in train_dl:
            x, y = batch[0].to(device), batch[1].to(device)
            mask = batch[2].to(device) if len(batch) > 2 else None

            opt.zero_grad()
            logits = net(x)
            loss = criterion(logits, y, mask)
            loss.backward()
            opt.step()
            running += loss.item()

        n = max(1, len(train_dl))
        print(f"epoch {epoch+1}/{args.epochs}  loss={running/n:.4f}")
        torch.save(net.state_dict(), args.ckpt)

    print(f"Saved weights to {args.ckpt}. "
          f"Run export_torchscript.py to produce the .pt for vol2bird.")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--ckpt", default="unet_radar_weights.pt")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
