"""Independent verification of output/release_v1.1 against v1.0 and COCO.

Rows are streamed in batches and v1.0 is held only as per-column SHA-1
digests, so the check fits in a few GB of memory rather than loading both
releases (about 11 GB of embedded PNGs) at once.
"""
import collections, glob, hashlib, json, os, tarfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2, pyarrow.parquet as pq

REL = "output/release_v1.1"
COCO = os.environ.get("COCO_ROOT", "/home/harshita/coco")
ann = json.load(open(f"{COCO}/annotations/instances_train2017.json"))
LIC = {im["id"]: im["license"] for im in ann["images"]}
LICTAB = {l["id"]: l for l in ann["licenses"]}
del ann
EXPECT = {"by": ({4, 7}, 2006), "by-nc": ({2}, 1697), "by-sa": ({5}, 1092), "by-nc-sa": ({1}, 3450)}
IMG = ("image", "trimap", "binary_mask", "hard_negatives")
PROV = ("coco_image_id", "coco_license_id", "license_name", "license_url", "flickr_url", "coco_url", "modified")
fail = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fail.append(msg)


def digest(value):
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha1(value).hexdigest()
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def batches(paths, batch_size=128):
    for f in paths:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_size):
            yield batch.to_pylist()


def decode_ok(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED) is not None


def audit(args):
    fname, img_b, tri_b, image_id = args
    forged = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.IMREAD_COLOR)
    tri = cv2.imdecode(np.frombuffer(tri_b, np.uint8), cv2.IMREAD_GRAYSCALE)
    orig = cv2.imread(f"{COCO}/train2017/{image_id:012d}.jpg", cv2.IMREAD_COLOR)
    if orig is None or orig.shape != forged.shape:
        return fname, None, None
    changed = np.abs(forged.astype(np.int16) - orig.astype(np.int16)).max(2) > 0
    target = tri == 255
    inter, union = (changed & target).sum(), (changed | target).sum()
    return fname, (inter / union if union else 1.0), int((changed & (tri == 0)).sum())


old_files = sorted(glob.glob("hf_export/*.parquet"))
old_cols = list(pq.read_schema(old_files[0]).names)
old = {}
for chunk in batches(old_files):
    for r in chunk:
        old[r["image_filename"]] = (r["image_id"], {c: digest(r[c]) for c in old_cols})
print(f"v1.0: {len(old)} rows hashed", flush=True)

new, per_cfg = {}, collections.Counter()
bad_lic, bad_prov, bad_cols, missing_gt, undecodable, results = [], 0, 0, 0, 0, []
with ProcessPoolExecutor(4) as ex:
    for cfg, (ids, n) in EXPECT.items():
        for chunk in batches(sorted(glob.glob(f"{REL}/data/{cfg}/*.parquet"))):
            jobs = []
            for r in chunk:
                per_cfg[cfg] += 1
                lic = LIC[r["image_id"]]
                if lic not in ids:
                    bad_lic.append((cfg, r["image_filename"], lic))
                new[r["image_filename"]] = r["image_id"]
                if not (all(r[c] not in (None, "") for c in PROV) and r["coco_image_id"] == r["image_id"]
                        and r["license_name"] == LICTAB[lic]["name"] and r["license_url"] == LICTAB[lic]["url"]
                        and r["modified"] is True):
                    bad_prov += 1
                ref = old.get(r["image_filename"])
                if ref is None or any(digest(r[c]) != ref[1][c] for c in old_cols):
                    bad_cols += 1
                if any(r[c] is None for c in IMG + ("patch_labels",)):
                    missing_gt += 1
                elif not all(decode_ok(r[c]["bytes"]) for c in IMG):
                    undecodable += 1
                jobs.append((r["image_filename"], r["image"]["bytes"], r["trimap"]["bytes"], r["image_id"]))
            results += list(ex.map(audit, jobs, chunksize=16))
        check(per_cfg[cfg] == n, f"{cfg}: {per_cfg[cfg]} rows (expected {n})")
        check(not [b for b in bad_lic if b[0] == cfg],
              f"{cfg}: every row's COCO licence (from annotations) is in {sorted(ids)}")

check(sum(per_cfg.values()) == 8245 and len(new) == 8245, f"total {len(new)} unique samples (expected 8245)")
check(not any(LIC[i] in (3, 6) for i in new.values()), "zero samples from licence 3 or 6 (checked by image_id against COCO)")
keep = {k for k, (iid, _) in old.items() if LIC[iid] not in (3, 6)}
check(set(new) == keep, "released set == v1.0 minus NoDerivs, by filename")
check(bad_prov == 0, f"licence and attribution fields present and match the COCO licences table ({bad_prov} bad)")
check(bad_cols == 0, f"all {len(old_cols)} v1.0 columns byte-identical for every released row ({bad_cols} differ)")
check(missing_gt == 0, "all four GT representations present (trimap, binary_mask, hard_negatives, patch_labels)")
check(undecodable == 0, "every image and mask decodes")

done = [x for x in results if x[1] is not None]
ious = np.array([x[1] for x in done])
outside = sum(x[2] for x in done)
print(f"pixel audit: {len(done)}/{len(results)} comparable; IoU mean {ious.mean():.4f} min {ious.min():.4f}; "
      f"pixels changed outside any mask: {outside}", flush=True)
check(outside == 0, "no pixel changed outside the labelled regions")

wkeys = collections.Counter()
for cfg in EXPECT:
    for t in sorted(glob.glob(f"{REL}/wds/{cfg}/*.tar")):
        with tarfile.open(t) as tf:
            for m in tf:
                if m.name.endswith(".json"):
                    j = json.load(tf.extractfile(m))
                    if LIC[j["image_id"]] not in EXPECT[cfg][0]:
                        fail.append(f"wds {cfg}: {m.name} has licence {LIC[j['image_id']]}")
                    wkeys[cfg] += 1
check(dict(wkeys) == dict(per_cfg), f"WebDataset per-config counts equal parquet: {dict(wkeys)}")
print("FAILED:" if fail else "ALL CHECKS PASSED", fail)
json.dump({"iou_mean": float(ious.mean()), "iou_min": float(ious.min()), "comparable": len(done),
           "outside": outside, "per_config": dict(per_cfg), "failed": fail},
          open(os.environ.get("VERIFY_OUT", "output/license_fix/verify_release.json"), "w"), indent=1)
