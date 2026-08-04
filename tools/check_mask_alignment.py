"""
Quality gate 1: mask ↔ pixel-diff alignment check.

For N random samples, diff the original COCO image against the
forged image and compare the changed-pixel set with the trimap's
target region. With inward-only feathering every changed pixel must
lie inside the labeled mask, so IoU should be ~1.0.

Rows with post-processing applied (Stage 2: JPEG/noise/blur/BC touch
every pixel) are excluded from the diff check and reported separately.

Usage:
    python tools/check_mask_alignment.py output/stage1_full \
        [--coco-dir ~/coco/train2017] [-n 20] [--seed 0] [--min-iou 0.97]

Exit code 0 = pass, 1 = fail.
"""

import argparse
import os
import sys


def _default_coco_dir():
    """COCO train2017 dir from $COCO_ROOT (default ~/coco)."""
    return os.path.join(
        os.environ.get("COCO_ROOT", os.path.expanduser("~/coco")),
        "train2017",
    )

import cv2
import numpy as np
import pandas as pd

PP_COLS = ["jpeg_quality", "noise_sigma", "blur", "contrast",
           "brightness"]


def row_is_postprocessed(row):
    for c in PP_COLS:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if c == "blur" and int(v) == 0:
            continue
        return True
    return False


def check(stage_dir, coco_dir, n, seed, min_iou):
    df = pd.read_csv(os.path.join(stage_dir, "metadata.csv"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))

    checked, skipped_pp, failed_load = 0, 0, 0
    ious, outside, inside_unchanged = [], [], []
    worst = []

    for i in idx:
        if checked >= n:
            break
        row = df.iloc[i]
        if row_is_postprocessed(row):
            skipped_pp += 1
            continue

        forged = cv2.imread(
            os.path.join(stage_dir, "tampered", row["image_filename"])
        )
        tri = cv2.imread(
            os.path.join(stage_dir, "masks", row["mask_filename"]),
            cv2.IMREAD_GRAYSCALE,
        )
        stem = os.path.splitext(row["image_filename"])[0]
        orig = cv2.imread(
            os.path.join(coco_dir, stem.split("_v")[0] + ".jpg")
        )
        if forged is None or tri is None or orig is None \
                or orig.shape != forged.shape:
            failed_load += 1
            continue

        vals = set(np.unique(tri).tolist())
        if not vals <= {0, 128, 255}:
            print(f"FAIL {row['image_filename']}: trimap has "
                  f"non-{{0,128,255}} values {sorted(vals)}")
            return 1

        diff = np.any(orig.astype(np.int16) != forged.astype(np.int16),
                      axis=2)
        tgt = tri == 255
        inter = np.logical_and(diff, tgt).sum()
        union = np.logical_or(diff, tgt).sum()
        iou = inter / max(union, 1)
        out_px = int(np.logical_and(diff, ~tgt).sum())
        in_unchanged = int(np.logical_and(tgt, ~diff).sum())

        ious.append(iou)
        outside.append(out_px)
        inside_unchanged.append(in_unchanged)
        worst.append((iou, row["image_filename"], out_px, in_unchanged))
        checked += 1

    if checked == 0:
        print("No checkable samples (all post-processed or missing).")
        return 0 if skipped_pp > 0 else 1

    worst.sort()
    print(f"Checked {checked} samples "
          f"(skipped {skipped_pp} post-processed, "
          f"{failed_load} load failures)")
    print(f"IoU(diff, target): mean={np.mean(ious):.4f} "
          f"min={np.min(ious):.4f}")
    print(f"Changed-outside-mask px: max={max(outside)} "
          f"mean={np.mean(outside):.1f}")
    print(f"In-mask-unchanged px:   max={max(inside_unchanged)} "
          f"mean={np.mean(inside_unchanged):.1f}")
    for iou, fname, out_px, in_un in worst[:3]:
        print(f"  worst: {fname} IoU={iou:.4f} "
              f"outside={out_px} unchanged={in_un}")

    ok = np.mean(ious) >= min_iou and max(outside) == 0
    print("PASS" if ok else
          f"FAIL (need mean IoU >= {min_iou} and 0 outside-mask px)")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage_dir")
    p.add_argument("--coco-dir", default=_default_coco_dir())
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-iou", type=float, default=0.97)
    a = p.parse_args()
    sys.exit(check(a.stage_dir, a.coco_dir, a.n, a.seed, a.min_iou))


if __name__ == "__main__":
    main()
