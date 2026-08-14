"""
Combined BCE + Dice loss for the 2-class (biological / non-biological)
per-elevation segmentation produced by unet.UNet.

The network outputs logits of shape (N, 2*n_elev, H, W) ordered elevation-major:
    [e0_nonbio, e0_bio, e1_nonbio, e1_bio, ...]

Targets are expected as one-hot of the same shape and ordering, with values in
{0, 1}. Pixels with no valid label (e.g. outside radar range) can be masked out
via an optional `mask` of shape (N, 1, H, W) or (N, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    loss = bce_weight * BCE(sigmoid logits, target)
         + dice_weight * (1 - soft Dice coefficient)

    Sigmoid (not softmax) is used so the two classes are treated as independent
    binary masks per elevation, which matches a "2 channels (bio + non-bio)"
    output where each channel is its own mask.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # pos_weight: optional per-channel tensor to counter class imbalance
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else None
        )

    def forward(self, logits, target, mask=None):
        # logits, target: (N, C, H, W)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)          # (N,1,H,W)
            mask = mask.to(logits.dtype)

        # ---- BCE term ----
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(),
            reduction="none",
            pos_weight=self.pos_weight,
        )
        if mask is not None:
            bce = bce * mask
            bce = bce.sum() / mask.expand_as(bce).sum().clamp_min(1.0)
        else:
            bce = bce.mean()

        # ---- Dice term ----
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask
            target = target * mask
        # reduce over spatial dims, keep per (N, C)
        dims = (2, 3)
        intersection = (probs * target).sum(dims)
        cardinality = probs.sum(dims) + target.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


if __name__ == "__main__":
    crit = BCEDiceLoss()
    logits = torch.randn(2, 10, 64, 64)
    target = (torch.rand(2, 10, 64, 64) > 0.5).float()
    print("loss:", crit(logits, target).item())
