# U-Net segmentation model for vol2bird

This replaces the original 3-class MistNet CNN segmentation model with a
**U-Net** that classifies each radar pixel as **biological** or
**non-biological**.

## What changed

### Python (this `training/` folder) — the model itself
The model architecture, training, and loss are **not** in the C code; vol2bird
only runs a serialized TorchScript `.pt`. These files produce that `.pt`:

| file | purpose |
|------|---------|
| `unet.py` | U-Net. Input **15 channels = 3 radar vars (DBZ, VRAD, WRAD) × 5 elevations**, output **2 classes per elevation** (non-bio, bio), 608×608. |
| `loss.py` | **BCE + Dice** loss (`BCEDiceLoss`), per-channel sigmoid. |
| `train.py` | Training loop skeleton — plug in your dataset in `RadarSegDataset`. |
| `export_torchscript.py` | Exports trained weights to a TorchScript `.pt` with the **class-major output layout** the C code requires, applying sigmoid so the C side gets probabilities. |

Workflow:
```bash
python train.py --data /path/to/data --epochs 50 --ckpt unet_radar_weights.pt
python export_torchscript.py --ckpt unet_radar_weights.pt --out unet_radar.pt
```
Then point vol2bird at it via `MISTNET_PATH` in your `options.conf`.

### C code — consume the 2-class output
- `lib/constants.h`: added `MISTNET_N_CLASS 2`, class indices
  `MISTNET_NONBIOLOGY_INDEX 0` / `MISTNET_BIOLOGY_INDEX 1`, and
  `MISTNET_BIOLOGY_THRESHOLD` (replaces the old 3-class weather indices/thresholds).
- `lib/librender.c`: output buffer sized to `MISTNET_N_CLASS` instead of 3;
  `addTensorToPolarVolume` / `addClassificationToPolarVolume` now derive the
  weather/clutter cell map from a **low biological probability** rule instead of
  a high weather-class rule. The `BACKGROUND` polar param is dropped; `WEATHER`
  now holds the non-biological probability and `BIOLOGY` the biological one.
- `libmistnet/libmistnet.cpp`: output copy loop sized `2×5×608×608`.

## Channel ordering contract (important)

**Input** (built in `librender.c::polarVolumeTo3DTensor`): channel index
`iElev + 5*iVar`, with `iVar` 0=DBZ, 1=VRAD, 2=WRAD →
`[dbz_e0..e4, vrad_e0..e4, wrad_e0..e4]`. Your training data must match.

**Output** (read in `librender.c` via `create4DTensor`): class-major
`[class][elev][x][y]` = `2×5×608×608`, class 0 = non-bio, class 1 = bio.
`export_torchscript.py` permutes the network's elevation-major output to this
layout, so do not change one side without the other.
