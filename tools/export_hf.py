"""
Package a generated stage directory as Hugging Face parquet shards.

Images, trimaps, binary masks and hard-negative masks are embedded as
PNG bytes (HF `Image` feature); patch labels are stored flattened with
their grid shape. Hard negatives are re-encoded from .npy to PNG —
lossless for binary masks and roughly two orders of magnitude smaller.

Usage:
    python tools/export_hf.py output/stage1_full hf_export \
        [--shard-mb 900] [--metadata metadata.csv]
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Columns that hold PNG bytes and should decode as HF `Image`.
IMAGE_COLS = ("image", "trimap", "binary_mask", "hard_negatives")

_ARROW_TO_HF = {
    "int8": "int8", "int16": "int16", "int32": "int32",
    "int64": "int64", "float": "float32", "double": "float64",
    "bool": "bool", "string": "string", "large_string": "string",
}


def hf_features(schema):
    """Build a datasets-compatible feature spec from an Arrow schema.

    Embedding this in the parquet key-value metadata is what makes
    `load_dataset` decode the image columns automatically (and the
    Hub dataset viewer render them) without a manual cast_column.
    """
    feats = {}
    for field in schema:
        name, t = field.name, field.type
        if name in IMAGE_COLS:
            feats[name] = {"_type": "Image"}
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            inner = _ARROW_TO_HF.get(str(t.value_type), "int64")
            feats[name] = {
                "feature": {"dtype": inner, "_type": "Value"},
                "_type": "Sequence",
            }
        else:
            feats[name] = {
                "dtype": _ARROW_TO_HF.get(str(t), "string"),
                "_type": "Value",
            }
    return feats


def with_hf_metadata(table):
    """Attach the datasets feature spec to a table's schema metadata."""
    meta = dict(table.schema.metadata or {})
    meta[b"huggingface"] = json.dumps(
        {"info": {"features": hf_features(table.schema)}}
    ).encode()
    return table.replace_schema_metadata(meta)


def fix_metadata(out_dir):
    """Re-stamp HF feature metadata onto already-written shards."""
    paths = sorted(glob.glob(os.path.join(out_dir, "*.parquet")))
    for p in paths:
        table = pq.read_table(p)
        pq.write_table(with_hf_metadata(table), p, compression="zstd")
        print(f"  stamped {p}")
    print(f"Done: {len(paths)} shard(s) carry HF feature metadata")
    return len(paths)


def _png_bytes(arr):
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def export(stage_dir, out_dir, shard_mb=900, metadata="metadata.csv"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(stage_dir, metadata))
    # Rows written before the column existed came from passes that
    # never reused a source object.
    if "source_reused" in df:
        df["source_reused"] = df["source_reused"].fillna(0).astype(int)

    meta_cols = [
        "image_id", "image_filename", "source_category_id",
        "source_category_name", "source_supercategory", "source_ann_id",
        "variant", "gate_profile", "source_reused", "transform_policy",
        "blend_mode", "flip", "scale_factor", "rotation_angle",
        "source_bbox", "target_bbox", "placement_tier", "horizon_tier",
        "same_cat_instance_count", "source_texture_var",
        "dest_texture_var", "bg_similarity_score",
        "image_height", "image_width",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]

    rows, shard_idx, nbytes, written = [], 0, 0, 0
    limit = shard_mb * 1024 * 1024

    def flush():
        nonlocal rows, shard_idx, nbytes
        if not rows:
            return
        table = with_hf_metadata(pa.Table.from_pylist(rows))
        path = os.path.join(
            out_dir, f"train-{shard_idx:04d}.parquet"
        )
        pq.write_table(table, path, compression="zstd")
        print(f"  wrote {path} ({len(rows)} rows, "
              f"{os.path.getsize(path) / 1e6:.0f} MB)")
        rows, nbytes = [], 0
        shard_idx += 1

    for _, r in df.iterrows():
        img_p = os.path.join(stage_dir, "tampered", r.image_filename)
        tri_p = os.path.join(stage_dir, "masks", r.mask_filename)
        bin_p = os.path.join(
            stage_dir, "binary_masks_flat", r.binary_mask_filename
        )
        pl_p = os.path.join(
            stage_dir, "patch_labels", r.patch_labels_filename
        )
        hn_p = os.path.join(
            stage_dir, "hard_negatives", r.hard_neg_filename
        )
        if not all(os.path.exists(p)
                   for p in (img_p, tri_p, bin_p, pl_p, hn_p)):
            print(f"  SKIP (missing files): {r.image_filename}")
            continue

        patch = np.load(pl_p)
        hard_neg = np.load(hn_p)

        rec = {c: (None if pd.isna(r[c]) else r[c]) for c in meta_cols}
        rec.update({
            "image": {"bytes": _read_bytes(img_p),
                      "path": r.image_filename},
            "trimap": {"bytes": _read_bytes(tri_p),
                       "path": r.mask_filename},
            "binary_mask": {"bytes": _read_bytes(bin_p),
                            "path": r.binary_mask_filename},
            "hard_negatives": {
                "bytes": _png_bytes((hard_neg > 0).astype(np.uint8) * 255),
                "path": r.hard_neg_filename.replace(".npy", ".png"),
            },
            "patch_labels": patch.astype(np.int8).flatten().tolist(),
            "patch_labels_shape": list(patch.shape),
        })
        rows.append(rec)
        nbytes += (
            len(rec["image"]["bytes"])
            + len(rec["trimap"]["bytes"])
            + len(rec["binary_mask"]["bytes"])
            + len(rec["hard_negatives"]["bytes"])
        )
        written += 1
        if nbytes >= limit:
            flush()
        if written % 1000 == 0:
            print(f"  ...{written}/{len(df)}")

    flush()
    print(f"Done: {written} rows → {shard_idx} parquet shard(s) "
          f"in {out_dir}")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage_dir")
    p.add_argument("out_dir")
    p.add_argument("--shard-mb", type=int, default=900)
    p.add_argument("--metadata", default="metadata.csv")
    p.add_argument(
        "--fix-metadata", action="store_true",
        help="Only re-stamp HF feature metadata on existing shards "
             "in out_dir (no re-export)",
    )
    a = p.parse_args()
    if a.fix_metadata:
        fix_metadata(a.out_dir)
    else:
        export(a.stage_dir, a.out_dir, a.shard_mb, a.metadata)


if __name__ == "__main__":
    main()
