"""
Quality gate 4: dataset validation — runnable by this repo AND the
training repo before consuming a generated stage.

Checks:
  * metadata.csv parses; no duplicate image filenames
  * every referenced file exists (tampered, trimap, binary, patch
    labels, hard negatives)
  * on a random pixel-check sample: trimap values ⊆ {0,128,255},
    binary values ⊆ {0,255}, non-zero source AND target areas,
    mask dims == image dims, patch-label grid shape and value range

Usage:
    python validate_dataset.py output/stage1_full [--sample 200]

Exit code 0 = pass, 1 = fail.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

SUBDIRS = {
    "image_filename": "tampered",
    "mask_filename": "masks",
    "binary_mask_filename": "binary_masks_flat",
    "patch_labels_filename": "patch_labels",
    "hard_neg_filename": "hard_negatives",
}


def validate(stage_dir, sample_n):
    errors = []
    meta_path = os.path.join(stage_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        print(f"FAIL: {meta_path} missing")
        return 1
    df = pd.read_csv(meta_path)
    print(f"metadata.csv: {len(df)} rows, {len(df.columns)} columns")

    dupes = df["image_filename"].duplicated().sum()
    if dupes:
        errors.append(f"{dupes} duplicate image filenames")

    # File existence for every row
    missing = 0
    for col, sub in SUBDIRS.items():
        for fname in df[col]:
            if not isinstance(fname, str):
                errors.append(f"non-string entry in {col}")
                continue
            if not os.path.exists(os.path.join(stage_dir, sub, fname)):
                missing += 1
                if missing <= 5:
                    errors.append(f"missing {sub}/{fname}")
    if missing > 5:
        errors.append(f"... {missing} missing files total")
    print(f"file existence: {missing} missing "
          f"of {len(df) * len(SUBDIRS)} referenced")

    # Pixel-level checks on a sample
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))[:min(sample_n, len(df))]
    for i in idx:
        row = df.iloc[i]
        tri = cv2.imread(
            os.path.join(stage_dir, "masks", row["mask_filename"]),
            cv2.IMREAD_GRAYSCALE,
        )
        img = cv2.imread(
            os.path.join(stage_dir, "tampered", row["image_filename"])
        )
        binm = cv2.imread(
            os.path.join(stage_dir, "binary_masks_flat",
                         row["binary_mask_filename"]),
            cv2.IMREAD_GRAYSCALE,
        )
        if tri is None or img is None or binm is None:
            errors.append(f"unreadable files for "
                          f"{row['image_filename']}")
            continue
        if tri.shape != img.shape[:2]:
            errors.append(f"mask/image dim mismatch: "
                          f"{row['image_filename']}")
        vals = set(np.unique(tri).tolist())
        if not vals <= {0, 128, 255}:
            errors.append(f"trimap values {sorted(vals)}: "
                          f"{row['mask_filename']}")
        bvals = set(np.unique(binm).tolist())
        if not bvals <= {0, 255}:
            errors.append(f"binary values {sorted(bvals)}: "
                          f"{row['binary_mask_filename']}")
        if (tri == 255).sum() == 0 or (tri == 128).sum() == 0:
            errors.append(f"zero-area source/target: "
                          f"{row['mask_filename']}")
        patches = np.load(
            os.path.join(stage_dir, "patch_labels",
                         row["patch_labels_filename"])
        )
        h, w = tri.shape
        exp = (-(-h // 16), -(-w // 16))  # ceil division
        if patches.shape != exp:
            errors.append(f"patch grid {patches.shape} != {exp}: "
                          f"{row['patch_labels_filename']}")
        if not set(np.unique(patches).tolist()) <= {-1, 0, 1, 2}:
            errors.append(f"bad patch label values: "
                          f"{row['patch_labels_filename']}")
    print(f"pixel checks: {len(idx)} samples")

    if errors:
        print(f"\nFAIL — {len(errors)} problems:")
        for e in errors[:30]:
            print(f"  {e}")
        return 1
    print("\nPASS")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage_dir")
    p.add_argument("--sample", type=int, default=200,
                   help="rows to pixel-check (default 200)")
    a = p.parse_args()
    sys.exit(validate(a.stage_dir, a.sample))


if __name__ == "__main__":
    main()
