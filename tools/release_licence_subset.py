"""
Cut a licence-compliant release from existing parquet and WebDataset shards.

MS-COCO images are Flickr photographs that keep their own Creative Commons
licences (the `license` field of each image in instances_train2017.json).
Each forgery is an adaptation of exactly one photograph, and a ShareAlike
licence requires adaptations to carry the same licence, so no single
licence covers the corpus. The release is instead a collection of
separately licensed parts, one config per compatible licence group:

    by         4 BY 2.0, 7 no known copyright restrictions
    by-nc      2 BY-NC 2.0
    by-sa      5 BY-SA 2.0
    by-nc-sa   1 BY-NC-SA 2.0
    (dropped)  3 BY-NC-ND 2.0, 6 BY-ND 2.0 -- no adaptations allowed

Each row carries its source photograph's licence and attribution fields,
read from the annotation file rather than from a table here. Rows are
copied byte-for-byte; only the dropped rows and the added columns differ
from the input shards.

Usage:
    python tools/release_licence_subset.py \
        --coco-ann /path/instances_train2017.json \
        --parquet-in hf_export --wds-in wds_export \
        --out output/release_v1.1
"""

import argparse
import glob
import io
import json
import os
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq

from export_hf import with_hf_metadata

CONFIG_BY_LICENSE_ID = {
    4: "by", 7: "by",
    2: "by-nc",
    5: "by-sa",
    1: "by-nc-sa",
}
CONFIGS = ("by", "by-nc", "by-sa", "by-nc-sa")

# Columns appended to every released row, in this order.
PROVENANCE_COLS = ("coco_image_id", "coco_license_id", "license_name",
                   "license_url", "flickr_url", "coco_url", "modified")


def load_provenance(coco_ann):
    with open(coco_ann) as f:
        d = json.load(f)
    lic = {l["id"]: l for l in d["licenses"]}
    prov = {}
    for im in d["images"]:
        L = lic[im["license"]]
        prov[im["id"]] = {
            "config": CONFIG_BY_LICENSE_ID.get(im["license"]),
            "coco_image_id": im["id"],
            "coco_license_id": im["license"],
            "license_name": L["name"],
            "license_url": L["url"],
            "flickr_url": im.get("flickr_url") or "",
            "coco_url": im.get("coco_url") or "",
            # Every sample is an adaptation of its photograph; BY-family
            # licences require saying so.
            "modified": True,
        }
    return prov


class _Sink:
    """Accumulate rows per config and flush them as numbered shards."""

    def __init__(self, out_dir, rows_per_shard):
        self.out_dir, self.limit = out_dir, rows_per_shard
        self.buf = {c: [] for c in CONFIGS}
        self.shard = {c: 0 for c in CONFIGS}
        self.count = {c: 0 for c in CONFIGS}

    def add(self, config, table):
        self.buf[config].append(table)
        if sum(t.num_rows for t in self.buf[config]) >= self.limit:
            self.flush(config)

    def flush(self, config):
        if not self.buf[config]:
            return
        table = pa.concat_tables(self.buf[config])
        d = os.path.join(self.out_dir, config)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"train-{self.shard[config]:04d}.parquet")
        pq.write_table(with_hf_metadata(table), path, compression="zstd")
        print(f"  wrote {path} ({table.num_rows} rows)")
        self.count[config] += table.num_rows
        self.shard[config] += 1
        self.buf[config] = []


def filter_parquet(src_dir, out_dir, prov, rows_per_shard=1100):
    sink = _Sink(out_dir, rows_per_shard)
    dropped = 0
    for path in sorted(glob.glob(os.path.join(src_dir, "*.parquet"))):
        table = pq.read_table(path)
        ids = [int(i) for i in table.column("image_id").to_pylist()]
        configs = [prov[i]["config"] for i in ids]
        dropped += configs.count(None)
        for config in CONFIGS:
            mask = pa.array([c == config for c in configs])
            part = table.filter(mask)
            if part.num_rows == 0:
                continue
            part_ids = [int(i) for i in part.column("image_id").to_pylist()]
            for col in PROVENANCE_COLS:
                part = part.append_column(
                    col, pa.array([prov[i][col] for i in part_ids])
                )
            sink.add(config, part)
    for config in CONFIGS:
        sink.flush(config)
    return sink.count, dropped


def filter_wds(src_dir, out_dir, prov, samples_per_shard=750):
    """Copy tar members of kept samples; rewrite only the .json member."""
    counts = {c: 0 for c in CONFIGS}
    shard = {c: 0 for c in CONFIGS}
    open_tars = {}
    dropped = 0

    def tar_for(config):
        t = open_tars.get(config)
        if t is not None and counts[config] % samples_per_shard == 0:
            t.close()
            shard[config] += 1
            t = None
        if t is None:
            d = os.path.join(out_dir, config)
            os.makedirs(d, exist_ok=True)
            t = tarfile.open(
                os.path.join(d, f"cmfd-train-{shard[config]:04d}.tar"), "w"
            )
            open_tars[config] = t
        return t

    for path in sorted(glob.glob(os.path.join(src_dir, "*.tar"))):
        with tarfile.open(path) as src:
            by_key = {}
            for m in src.getmembers():
                by_key.setdefault(m.name.split(".", 1)[0], []).append(m)
            for key, ms in by_key.items():
                meta_m = next(m for m in ms if m.name.endswith(".json"))
                meta = json.load(src.extractfile(meta_m))
                p = prov[int(meta["image_id"])]
                if p["config"] is None:
                    dropped += 1
                    continue
                dst = tar_for(p["config"])
                for m in ms:
                    if m.name.endswith(".json"):
                        meta.update({c: p[c] for c in PROVENANCE_COLS})
                        data = json.dumps(meta).encode()
                        info = tarfile.TarInfo(m.name)
                        info.size, info.mtime = len(data), m.mtime
                        dst.addfile(info, io.BytesIO(data))
                    else:
                        dst.addfile(m, src.extractfile(m))
                counts[p["config"]] += 1
    for t in open_tars.values():
        t.close()
    return counts, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-ann", required=True)
    ap.add_argument("--parquet-in", default="hf_export")
    ap.add_argument("--wds-in", default="wds_export")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prov = load_provenance(a.coco_ann)
    pk, pdrop = filter_parquet(a.parquet_in, os.path.join(a.out, "data"),
                               prov)
    wk, wdrop = filter_wds(a.wds_in, os.path.join(a.out, "wds"), prov)
    for c in CONFIGS:
        print(f"{c:<9} parquet {pk[c]:>5}  wds {wk[c]:>5}")
    print(f"dropped   parquet {pdrop:>5}  wds {wdrop:>5}")
    if (pk, pdrop) != (wk, wdrop):
        raise SystemExit("parquet and WebDataset disagree")


if __name__ == "__main__":
    main()
