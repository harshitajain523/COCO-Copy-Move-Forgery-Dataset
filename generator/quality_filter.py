"""
Post-hoc quality filter for already-generated COCO-CMFD samples.

Audits every row of a stage's metadata.csv against the same
plausibility rules the (fixed) generator now enforces at creation
time, so datasets generated with older code can be pruned instead
of regenerated:

  mirrored_text          — flipped copy of a text-bearing category
  unconstrained_height   — ground object placed with no vertical
                           constraint (horizon_tier == permissive)
  perspective_mismatch   — applied scale contradicts the depth
                           implied by the destination height
  support_violation      — pasted object's footprint rests on an
                           incompatible surface (wall/sky under a
                           floor-standing object)
  mask_fills_bbox        — organic object whose mask ~fills its
                           bbox (segmentation leaked background)

Usage:
    python generator/quality_filter.py output/stage1_clean
    python generator/quality_filter.py output/stage1_clean --prune

Without --prune only reports and writes metadata_clean.csv +
rejected.csv. With --prune additionally moves rejected images,
masks, and auxiliary files into <stage_dir>/rejected/.
"""

import argparse
import ast
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import BOXY_CATS, GROUND_CATS, NO_FLIP_CATS  # noqa: E402
from scene_layout import (  # noqa: E402
    build_surface_map,
    is_support_compatible,
    support_surface_code,
)


def _img_id_from_filename(fname):
    """COCO filenames encode the image id: 000000403571.png → 403571.

    Variant suffixes (000000403571_v1.png) are stripped first.
    """
    stem = os.path.splitext(fname)[0]
    stem = stem.split("_v")[0]
    return int(stem)


def _target_mask(trimap):
    return (trimap == 255).astype(np.uint8)


def _source_mask(trimap):
    return (trimap == 128).astype(np.uint8)


def check_row(row, stage_dir, stuff_coco, surface_cache):
    """Return a list of failure reasons for one metadata row."""
    reasons = []
    cat = int(row["source_category_id"])
    h_img = int(row["image_height"])

    # 1. Mirrored text
    if int(row.get("flip", 0)) == 1 and cat in NO_FLIP_CATS:
        reasons.append("mirrored_text")

    src_x, src_y, src_w, src_h = ast.literal_eval(row["source_bbox"])
    dst_x, dst_y, dst_w, dst_h = ast.literal_eval(row["target_bbox"])
    src_cy = src_y + src_h / 2.0
    dst_cy = dst_y + dst_h / 2.0

    # 2. Perspective consistency (scale vs implied depth)
    expected = 1.0 + 0.5 * ((dst_cy - src_cy) / max(h_img, 1))
    ratio = float(row["scale_factor"]) / max(expected, 1e-6)
    if ratio < 0.80 or ratio > 1.25:
        reasons.append("perspective_mismatch")

    # 3 + 4 + 5 need the trimap
    tri_path = os.path.join(stage_dir, "masks", row["mask_filename"])
    trimap = cv2.imread(tri_path, cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        reasons.append("missing_mask")
        return reasons

    tgt = _target_mask(trimap)
    src = _source_mask(trimap)

    # 3. Support-surface violation
    support_confirmed = False
    if stuff_coco is not None and src.any() and tgt.any():
        # Prefer the explicit metadata column (added 2026-07-07);
        # fall back to filename parsing for older datasets.
        img_id = row.get("image_id")
        if img_id is None or pd.isna(img_id):
            img_id = _img_id_from_filename(row["image_filename"])
        img_id = int(img_id)
        if img_id not in surface_cache:
            surface_cache[img_id] = build_surface_map(
                stuff_coco, img_id, trimap.shape[:2]
            )
        smap = surface_cache[img_id]
        src_support = support_surface_code(smap, src)
        dst_support = support_surface_code(smap, tgt)
        if src_support > 0 and dst_support > 0:
            support_confirmed = is_support_compatible(
                src_support, dst_support
            )
            if not support_confirmed:
                reasons.append("support_violation")
        elif (
            dst_support == 0
            and src_support == 1
            and cat in GROUND_CATS
            and dst_cy < 0.55 * h_img
        ):
            reasons.append("support_violation")

    # 4. Ground object with unconstrained vertical placement AND no
    # confirmed support surface — placement height was pure luck.
    if (
        row.get("horizon_tier") == "permissive"
        and cat in GROUND_CATS
        and not support_confirmed
    ):
        reasons.append("unconstrained_height")

    # 5. Mask ≈ bbox for organic categories
    if tgt.any() and cat not in BOXY_CATS:
        ys, xs = np.where(tgt > 0)
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if tgt.sum() / max(bbox_area, 1) > 0.92:
            reasons.append("mask_fills_bbox")

    return reasons


def run_filter(stage_dir, prune=False):
    meta_path = os.path.join(stage_dir, "metadata.csv")
    df = pd.read_csv(meta_path)

    # Sharded full-COCO runs emit stuff_subset_shardXofY.json per
    # shard. Load ONE stuff file at a time (memory-safe) and check
    # only the rows whose image belongs to it.
    stuff_files = sorted(
        glob.glob(os.path.join(stage_dir, "stuff_subset*.json"))
    )
    if not stuff_files:
        print("WARN: no stuff_subset*.json — support checks skipped")

    row_img_ids = {}
    for idx, row in df.iterrows():
        img_id = row.get("image_id")
        if img_id is None or pd.isna(img_id):
            img_id = _img_id_from_filename(row["image_filename"])
        row_img_ids[idx] = int(img_id)

    verdicts = {}
    remaining = set(df.index)
    for stuff_path in stuff_files:
        stuff_coco = COCO(stuff_path)
        covered = set(stuff_coco.getImgIds())
        surface_cache = {}
        for idx in sorted(remaining):
            if row_img_ids[idx] not in covered:
                continue
            reasons = check_row(
                df.loc[idx], stage_dir, stuff_coco, surface_cache
            )
            verdicts[idx] = ";".join(reasons)
            remaining.discard(idx)
            # Keep the cache bounded (surface maps are H×W arrays)
            if len(surface_cache) > 64:
                surface_cache.clear()
        del stuff_coco, surface_cache

    # Rows with no stuff coverage: run the stuff-independent checks.
    for idx in sorted(remaining):
        reasons = check_row(df.loc[idx], stage_dir, None, {})
        verdicts[idx] = ";".join(reasons)

    df["reject_reasons"] = [verdicts[idx] for idx in df.index]
    clean = df[df["reject_reasons"] == ""].drop(columns=["reject_reasons"])
    rejected = df[df["reject_reasons"] != ""]

    clean.to_csv(os.path.join(stage_dir, "metadata_clean.csv"), index=False)
    rejected.to_csv(os.path.join(stage_dir, "rejected.csv"), index=False)

    print(f"Total: {len(df)}  Clean: {len(clean)}  "
          f"Rejected: {len(rejected)}")
    if len(rejected):
        counts = (
            rejected["reject_reasons"].str.split(";").explode()
            .value_counts()
        )
        print("Rejection breakdown:")
        for reason, n in counts.items():
            print(f"  {n:5d} × {reason}")

    if prune and len(rejected):
        rej_dir = os.path.join(stage_dir, "rejected")
        os.makedirs(rej_dir, exist_ok=True)
        subdirs = {
            "image_filename": "tampered",
            "mask_filename": "masks",
            "binary_mask_filename": "binary_masks_flat",
            "patch_labels_filename": "patch_labels",
            "hard_neg_filename": "hard_negatives",
        }
        moved = 0
        for _, row in rejected.iterrows():
            for col, sub in subdirs.items():
                fname = row.get(col)
                if not isinstance(fname, str):
                    continue
                src = os.path.join(stage_dir, sub, fname)
                if os.path.exists(src):
                    os.makedirs(os.path.join(rej_dir, sub), exist_ok=True)
                    shutil.move(src, os.path.join(rej_dir, sub, fname))
                    moved += 1
        print(f"Moved {moved} files to {rej_dir}/")

    return clean, rejected


def main():
    parser = argparse.ArgumentParser(
        description="Audit generated CMFD samples for plausibility",
    )
    parser.add_argument("stage_dir", help="e.g. output/stage1_clean")
    parser.add_argument(
        "--prune", action="store_true",
        help="Move rejected samples into <stage_dir>/rejected/",
    )
    args = parser.parse_args()
    run_filter(args.stage_dir, prune=args.prune)


if __name__ == "__main__":
    main()
