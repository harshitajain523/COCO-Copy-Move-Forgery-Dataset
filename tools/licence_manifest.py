"""
Build the per-sample licence manifest for a released parquet corpus.

Every licence name and URL is read from the `licenses` table of the COCO
annotation file, and every sample's licence from its source image's
`license` field, so nothing here depends on a hand-copied table.

Columns:
    sample_id           image_filename without extension (unique per row)
    coco_image_id       source photograph
    coco_license_id     COCO licence id of that photograph
    license_name        its licence, as named by COCO
    license_url         its licence URL, as given by COCO
    flickr_url          the photograph on Flickr
    coco_url            the photograph on the COCO server
    split               split in the released corpus
    training_split      split used by the training repository, if any
    released            False when the licence forbids derivatives

Usage:
    python tools/licence_manifest.py \
        --coco-ann /path/instances_train2017.json \
        --parquet hf_export --out output/license_fix/license_manifest.csv \
        [--training-splits m3_split_assignment.csv]
"""

import argparse
import csv
import glob
import json
import os

import pyarrow.parquet as pq

NODERIVS_LICENSE_IDS = frozenset({3, 6})


def build(coco_ann, parquet_dir, training_splits=None):
    with open(coco_ann) as f:
        d = json.load(f)
    lic = {l["id"]: l for l in d["licenses"]}
    img = {im["id"]: im for im in d["images"]}

    splits = {}
    if training_splits:
        with open(training_splits) as f:
            splits = {r["sample_id"]: r["fixed_split"]
                      for r in csv.DictReader(f)}

    rows = []
    for path in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        for r in pq.read_table(
            path, columns=["image_filename", "image_id"]
        ).to_pylist():
            im = img[int(r["image_id"])]
            L = lic[im["license"]]
            sid = os.path.splitext(r["image_filename"])[0]
            rows.append({
                "sample_id": sid,
                "coco_image_id": im["id"],
                "coco_license_id": im["license"],
                "license_name": L["name"],
                "license_url": L["url"],
                "flickr_url": im.get("flickr_url", ""),
                "coco_url": im.get("coco_url", ""),
                "split": "train",
                "training_split": splits.get(sid, ""),
                "released": im["license"] not in NODERIVS_LICENSE_IDS,
            })
    return rows, lic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-ann", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--training-splits")
    a = ap.parse_args()

    rows, lic = build(a.coco_ann, a.parquet, a.training_splits)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows")
    for i in sorted(lic):
        n = sum(r["coco_license_id"] == i for r in rows)
        print(f"  {i}  {lic[i]['name']:<45} {lic[i]['url']:<50} {n}")


if __name__ == "__main__":
    main()
