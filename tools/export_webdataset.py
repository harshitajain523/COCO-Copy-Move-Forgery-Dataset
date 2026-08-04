"""
Package a generated stage directory as WebDataset tar shards.

One sample per key, files kept byte-identical to the generated
dataset (no re-encoding). Standard library only.

Members per sample:
    <key>.png            forged image
    <key>.trimap.png     0=background, 128=source, 255=target
    <key>.binary.png     0/255 union mask
    <key>.hardneg.npy    non-source COCO objects (uint8 0/1)
    <key>.patches.npy    16px patch-grid labels (int8, -1/0/1/2)
    <key>.json           per-sample metadata

Usage:
    python tools/export_webdataset.py output/stage1_full wds_export \
        [--shard-mb 900] [--metadata metadata.csv]
"""

import argparse
import json
import os
import tarfile

import pandas as pd


META_COLS = [
    "image_id", "image_filename", "source_category_id",
    "source_category_name", "source_supercategory", "source_ann_id",
    "variant", "gate_profile", "source_reused", "transform_policy",
    "blend_mode", "flip", "scale_factor", "rotation_angle",
    "source_bbox", "target_bbox", "placement_tier", "horizon_tier",
    "same_cat_instance_count", "source_texture_var",
    "dest_texture_var", "bg_similarity_score",
    "image_height", "image_width",
]


def export(stage_dir, out_dir, shard_mb=900, metadata="metadata.csv"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(stage_dir, metadata))
    if "source_reused" in df:
        df["source_reused"] = df["source_reused"].fillna(0).astype(int)
    cols = [c for c in META_COLS if c in df.columns]

    limit = shard_mb * 1024 * 1024
    shard_idx, written = 0, 0
    tar = None
    tar_path = None

    def open_shard():
        nonlocal tar, tar_path, shard_idx
        tar_path = os.path.join(
            out_dir, f"cmfd-train-{shard_idx:04d}.tar"
        )
        tar = tarfile.open(tar_path, "w")

    def close_shard():
        nonlocal tar, shard_idx
        if tar is None:
            return
        tar.close()
        print(f"  wrote {tar_path} "
              f"({os.path.getsize(tar_path) / 1e6:.0f} MB)")
        tar = None
        shard_idx += 1

    open_shard()
    for _, r in df.iterrows():
        key = os.path.splitext(r.image_filename)[0]
        members = [
            (f"{key}.png",
             os.path.join(stage_dir, "tampered", r.image_filename)),
            (f"{key}.trimap.png",
             os.path.join(stage_dir, "masks", r.mask_filename)),
            (f"{key}.binary.png",
             os.path.join(stage_dir, "binary_masks_flat",
                          r.binary_mask_filename)),
            (f"{key}.hardneg.npy",
             os.path.join(stage_dir, "hard_negatives",
                          r.hard_neg_filename)),
            (f"{key}.patches.npy",
             os.path.join(stage_dir, "patch_labels",
                          r.patch_labels_filename)),
        ]
        if not all(os.path.exists(p) for _, p in members):
            print(f"  SKIP (missing files): {r.image_filename}")
            continue

        for arcname, path in members:
            tar.add(path, arcname=arcname)

        rec = {c: (None if pd.isna(r[c]) else r[c]) for c in cols}
        blob = json.dumps(rec, default=str).encode()
        info = tarfile.TarInfo(f"{key}.json")
        info.size = len(blob)
        import io as _io
        tar.addfile(info, _io.BytesIO(blob))

        written += 1
        if written % 1000 == 0:
            print(f"  ...{written}/{len(df)}")
        if os.path.getsize(tar_path) >= limit:
            close_shard()
            open_shard()

    close_shard()
    print(f"Done: {written} samples → {shard_idx} tar shard(s) "
          f"in {out_dir}")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage_dir")
    p.add_argument("out_dir")
    p.add_argument("--shard-mb", type=int, default=900)
    p.add_argument("--metadata", default="metadata.csv")
    a = p.parse_args()
    export(a.stage_dir, a.out_dir, a.shard_mb, a.metadata)


if __name__ == "__main__":
    main()
