"""
Entry point for COCO-CMFD dataset generation pipeline.

STAGE-AWARE VERSION
===================
Supports two generation modes:
  Stage 1 (clean):     Zero post-processing. Pure copy signal
                       for contrastive pretraining.
  Stage 2 (synthetic): Full post-processing suite. DeFaCTo-like
                       synthetic complexity.

Key design decisions
--------------------
* Annotation pre-filter via subprocess → main process never loads
  the 449 MB file.
* Optional stuff-annotation loading (also subprocess-isolated).
* Single-process only — multiprocessing removed entirely.
* Deterministic per-(image, variant) RNG seeding.
* Append-only resume ledger (completed.jsonl) + incremental skip log.
* Sharding (--shard/--num-shards) for full-COCO runs.
* gc.collect() after every image.
* resource.setrlimit as a safety net.
"""

import dataclasses
import datetime
import gc
import json
import logging
import os
import random
import resource
import subprocess
import sys
import traceback

import cv2
import pandas as pd
from pycocotools.coco import COCO

# Ensure generator/ is on the import path when run from project root
sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)

from generator import CopyMoveGenerator  # noqa: E402
from stage_config import get_stage_config  # noqa: E402

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────
# COCO location resolves in this order:
#   1. --coco-root CLI argument
#   2. COCO_ROOT environment variable
#   3. ~/coco
# Output defaults to <project>/output, overridable via --output-dir
# or the CMFD_OUTPUT environment variable.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DEFAULT_COCO_ROOT = os.environ.get(
    "COCO_ROOT", os.path.expanduser("~/coco")
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "CMFD_OUTPUT", os.path.join(_PROJECT_ROOT, "output")
)


def coco_paths(coco_root=None):
    """Resolve (images_dir, instances_json, stuff_json) for a root."""
    root = coco_root or DEFAULT_COCO_ROOT
    return (
        os.path.join(root, "train2017"),
        os.path.join(root, "annotations", "instances_train2017.json"),
        os.path.join(root, "annotations", "stuff_train2017.json"),
    )

# ── Tunables ─────────────────────────────────────────────────────
SEED = 1337
MEM_LIMIT_SOFT_GB = 5.0
MEM_LIMIT_HARD_GB = 6.0


# ── Helpers ──────────────────────────────────────────────────────
def _apply_memory_limit(soft_gb=5.0, hard_gb=6.0):
    """Set a virtual-memory ceiling so we crash cleanly, not the OS."""
    try:
        soft = int(soft_gb * 1024 ** 3)
        hard = int(hard_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        logger.info(
            "Memory limit: %.1f GB soft / %.1f GB hard",
            soft_gb, hard_gb,
        )
    except Exception as exc:
        logger.warning("Could not set memory limit: %s", exc)


def _create_subset_annotations(annot_path, subset_path,
                               n_images=200, seed=1337,
                               shard=0, num_shards=1):
    """
    Create a small annotation subset using a subprocess.

    The subprocess loads the full 449 MB JSON in its own address
    space, filters it down, writes the result, and exits —
    returning ALL RAM to the OS.

    A sidecar (<subset>.meta) records the parameters used; a cached
    subset is only reused when they match, so changing --seed or the
    subset size can never silently serve stale sampling.
    """
    params = {"n_images": n_images, "seed": seed,
              "shard": shard, "num_shards": num_shards}
    sidecar = subset_path + ".meta"
    if os.path.exists(subset_path):
        try:
            with open(sidecar) as f:
                if json.load(f) == params:
                    logger.info(
                        "Reusing cached subset: %s", subset_path
                    )
                    return
        except (OSError, json.JSONDecodeError):
            pass
        logger.info(
            "Cached subset params changed — regenerating %s",
            subset_path,
        )

    logger.info(
        "Creating annotation subset (%s images, shard %d/%d) "
        "via subprocess...",
        n_images if n_images > 0 else "all", shard, num_shards,
    )

    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "filter_annotations.py",
    )
    result = subprocess.run(
        [
            sys.executable, script,
            annot_path, subset_path,
            str(n_images), str(seed),
            str(shard), str(num_shards),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.error("Subset creation FAILED:\n%s", result.stderr)
        sys.exit(1)

    for line in result.stdout.strip().splitlines():
        logger.info("[subset] %s", line)

    with open(sidecar, "w") as f:
        json.dump(params, f)


def _create_stuff_subset(stuff_path, stuff_subset_path,
                         image_ids_path):
    """
    Create a stuff annotation subset using a subprocess.

    Only runs if stuff_train2017.json exists on disk.
    """
    if not os.path.exists(stuff_path):
        logger.info(
            "Stuff annotations not found at %s — using "
            "heuristic fallback for surface placement.",
            stuff_path,
        )
        return False

    if os.path.exists(stuff_subset_path):
        logger.info(
            "Reusing cached stuff subset: %s",
            stuff_subset_path,
        )
        return True

    logger.info("Creating stuff annotation subset via subprocess...")

    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "filter_stuff.py",
    )
    result = subprocess.run(
        [
            sys.executable, script,
            stuff_path, stuff_subset_path, image_ids_path,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.warning(
            "Stuff subset creation failed (non-fatal):\n%s",
            result.stderr,
        )
        return False

    for line in result.stdout.strip().splitlines():
        logger.info("[stuff] %s", line)

    return True


# NOTE: the old _maybe_downscale() helper was removed on purpose
# (2026-07-07): it resized the image without rescaling the COCO
# annotations, which would silently misalign every mask. Images are
# always processed at native resolution now.


# ── Stage-aware generation ───────────────────────────────────────
def run_stage(stage_name, num_images=None, seed=SEED,
              output_dir=None, shard=0, num_shards=1,
              subset_size=None, retry_failed=False,
              extend_variants=False, reuse_sources=False,
              coco_root=None):
    """
    Run dataset generation for a given stage.

    Parameters
    ----------
    stage_name : str
        'stage1' or 'stage2'.
    num_images : int, optional
        Override the default number of images (global target —
        shared across shards/resumes via the ledger).
    seed : int
        Random seed for reproducibility.
    shard, num_shards : int
        Partition of the full COCO train set (num_shards > 1
        ignores subset_size and processes shard `shard`).
    subset_size : int, optional
        Override the preset's annotation-subset size.
    """
    overrides = {}
    if num_images is not None:
        overrides["num_images"] = num_images
    if subset_size is not None:
        overrides["subset_size"] = subset_size
    cfg = get_stage_config(stage_name, **overrides)

    logger.info("=" * 60)
    logger.info("STAGE: %s", cfg.name)
    logger.info("Description: %s", cfg.description)
    logger.info("Target images: %d", cfg.num_images)
    logger.info("Post-processing: JPEG=%.0f%% Noise=%.0f%% "
                "Blur=%.0f%% BC=%.0f%%",
                cfg.jpeg_prob * 100, cfg.noise_prob * 100,
                cfg.blur_prob * 100, cfg.bc_prob * 100)
    logger.info("=" * 60)

    # Safety first
    _apply_memory_limit(MEM_LIMIT_SOFT_GB, MEM_LIMIT_HARD_GB)
    cv2.setNumThreads(1)

    coco_images_dir, annot_path, stuff_path = coco_paths(coco_root)
    if not os.path.isdir(coco_images_dir):
        logger.error(
            "COCO images not found at %s — set --coco-root or the "
            "COCO_ROOT environment variable.", coco_images_dir,
        )
        sys.exit(1)

    # Output directories (stage-specific)
    stage_output = output_dir or os.path.join(
        DEFAULT_OUTPUT_DIR, cfg.name
    )
    output_tampered = os.path.join(stage_output, "tampered")
    output_masks = os.path.join(stage_output, "masks")
    output_logs = os.path.join(stage_output, "logs")
    for d in (output_tampered, output_masks, output_logs):
        os.makedirs(d, exist_ok=True)

    # ── Step 1: annotation subset (subprocess) ──────────────────
    # Sharded runs keep per-shard subset files so only ~1/num_shards
    # of COCO is ever indexed in this process at a time.
    shard_tag = (
        f"_shard{shard}of{num_shards}" if num_shards > 1 else ""
    )
    subset_path = os.path.join(
        stage_output, f"annotations_subset{shard_tag}.json"
    )
    _create_subset_annotations(
        annot_path, subset_path,
        n_images=cfg.subset_size if num_shards == 1 else 0,
        seed=seed, shard=shard, num_shards=num_shards,
    )

    # ── Step 2: load instance annotations ────────────────────────
    logger.info("Loading annotation subset...")
    coco = COCO(subset_path)
    img_ids = coco.getImgIds()
    logger.info("Subset contains %d images", len(img_ids))

    # ── Step 3: stuff annotations (optional, subprocess) ─────────
    stuff_coco = None
    if cfg.use_stuff_annotations:
        # Save image IDs for the stuff filter subprocess
        ids_path = os.path.join(
            stage_output, f"image_ids{shard_tag}.json"
        )
        with open(ids_path, "w") as f:
            json.dump(img_ids, f)

        stuff_subset_path = os.path.join(
            stage_output, f"stuff_subset{shard_tag}.json"
        )
        has_stuff = _create_stuff_subset(
            stuff_path, stuff_subset_path, ids_path,
        )
        if has_stuff:
            logger.info("Loading stuff annotation subset...")
            stuff_coco = COCO(stuff_subset_path)
            logger.info(
                "Stuff subset: %d annotations",
                len(stuff_coco.getAnnIds()),
            )

    # ── Step 4: create generator ─────────────────────────────────
    gen_kwargs = cfg.to_generator_kwargs(
        output_dir_tampered=output_tampered,
        output_dir_masks=output_masks,
    )
    generator = CopyMoveGenerator(**gen_kwargs)

    # ── Step 5: generate forgeries ───────────────────────────────
    logger.info(
        "Generating %d copy-move forgeries...", cfg.num_images
    )

    rng = random.Random(seed)
    rng.shuffle(img_ids)

    # ── Resume ledger ────────────────────────────────────────────
    # One JSON line per generated sample, flushed immediately. On
    # restart (or on the next shard of a full run) completed images
    # are skipped, and metadata.csv is rebuilt from the ledger so a
    # crash can never lose provenance for files already on disk.
    ledger_path = os.path.join(stage_output, "completed.jsonl")
    ledger_rows = []
    done_img_ids = set()
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line from a crash
                ledger_rows.append(row)
                done_img_ids.add(row.get("image_id"))
        logger.info(
            "Resume ledger: %d samples from %d images already done",
            len(ledger_rows), len(done_img_ids),
        )

    # Previously-failed images are skipped on resume: per-image
    # seeding makes failures deterministic, so re-attempting them
    # under the SAME gates can only burn time. In --retry-failed
    # (recovery) mode, images whose failure reasons the relaxed
    # gates can address are retried instead; recovery failures go
    # to their own log so an interrupted recovery also resumes
    # without re-walking.
    strict_log = os.path.join(output_logs, "skips_incremental.log")
    recovery_log = os.path.join(
        output_logs, "skips_incremental_recovery.log"
    )
    extend_log = os.path.join(
        output_logs,
        "skips_incremental_extend2.log" if reuse_sources
        else "skips_incremental_extend.log",
    )
    if extend_variants:
        skiplog_path = extend_log
    elif retry_failed:
        skiplog_path = recovery_log
    else:
        skiplog_path = strict_log

    def _read_skiplog(path):
        reasons = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.split("\t", 1)
                    try:
                        iid = int(parts[0])
                    except (ValueError, IndexError):
                        continue
                    reasons.setdefault(iid, set()).add(
                        parts[1].strip() if len(parts) > 1 else ""
                    )
        return reasons

    RECOVERABLE = (
        "patch_extract_failed", "object_not_isolated",
        "no_valid_destination", "tamper_too_small",
        "low_boundary_quality", "source_target_overlap_too_high",
        # recoverable since the relaxed_v2 paste-size floor: images
        # whose only objects were between 1.2% and 2% of image area
        "no_suitable_annotations",
    )
    failed_img_ids = set()
    if extend_variants:
        # Variant extension: revisit only SUCCESSFUL images (proven
        # to have valid placements) and generate additional variants
        # from source objects not yet used. Images already extended
        # (extend log) are skipped on resume.
        failed_img_ids = set(_read_skiplog(extend_log))
        logger.info(
            "Extension mode: %d successful images in pool "
            "(%d already extended)",
            len(done_img_ids), len(failed_img_ids),
        )
    elif retry_failed:
        strict_reasons = _read_skiplog(strict_log)
        for iid, rs in strict_reasons.items():
            if not any(
                any(k in r for k in RECOVERABLE) for r in rs
            ):
                failed_img_ids.add(iid)  # immutable failure
        failed_img_ids |= set(_read_skiplog(recovery_log))
        logger.info(
            "Recovery mode: %d images eligible for retry "
            "(%d skipped as immutable/already-retried)",
            len(strict_reasons) - len(failed_img_ids),
            len(failed_img_ids),
        )
        failed_img_ids -= done_img_ids
    else:
        failed_img_ids |= set(_read_skiplog(strict_log))
        failed_img_ids |= set(_read_skiplog(recovery_log))
        if failed_img_ids:
            logger.info(
                "Skip log: %d previously-failed images will be "
                "skipped on resume", len(failed_img_ids),
            )
        failed_img_ids -= done_img_ids

    # Per-image ledger state for extension mode: which source
    # objects are already used, and the highest variant index.
    used_by_img = {}
    maxv_by_img = {}
    if extend_variants:
        for row in ledger_rows:
            iid = row.get("image_id")
            used_by_img.setdefault(iid, set()).add(
                row.get("source_ann_id", -1)
            )
            try:
                v = int(str(row.get("variant", "_v0")).lstrip("_v"))
            except ValueError:
                v = 0
            maxv_by_img[iid] = max(maxv_by_img.get(iid, -1), v)
    # Hard ceiling on variants per image across all passes.
    MAX_VARIANT_IDX = 8

    successes = list(ledger_rows)
    new_successes = 0
    failures = []

    # Per-category quota: no category may exceed this share of the
    # target. Counts include ledger rows so the cap holds across
    # resumes and shards.
    cat_cap = max(3, int(cfg.category_share_cap * cfg.num_images))
    cat_counts = {}
    for row in ledger_rows:
        cat = row.get("source_category_id", -1)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    ledger_f = open(ledger_path, "a")
    skips_f = open(skiplog_path, "a")

    for img_id in img_ids:
        if len(successes) >= cfg.num_images:
            break
        if extend_variants:
            # Only successful, not-yet-extended images
            if img_id not in done_img_ids \
                    or img_id in failed_img_ids:
                continue
        elif img_id in done_img_ids or img_id in failed_img_ids:
            continue

        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(coco_images_dir, file_name)

        # Cheap pre-filter from COCO metadata: reject small images
        # without paying for an imread.
        if min(
            int(img_info.get("height", 0)),
            int(img_info.get("width", 0)),
        ) < cfg.min_resolution:
            failures.append("SKIP: low_resolution")
            skips_f.write(f"{img_id}\tSKIP: low_resolution\n")
            continue

        # Annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        # Multiple variants per image, each from a different source
        # object. Image-level failures end the image; object- or
        # placement-level failures still let later variants try a
        # different object. Extension mode continues from the next
        # unused variant index and excludes already-used sources.
        if extend_variants:
            base_v = maxv_by_img.get(img_id, -1) + 1
            # Re-paste mode deliberately allows proven source
            # objects again (new destination, new sample); rows are
            # marked via the source_reused metadata column below.
            used_ann_ids = (
                set() if reuse_sources
                else set(used_by_img.get(img_id, ()))
            )
        else:
            base_v = 0
            used_ann_ids = set()
        for k in range(max(1, cfg.variants_per_image)):
            variant = base_v + k
            if variant >= MAX_VARIANT_IDX:
                break
            if len(successes) >= cfg.num_images:
                break

            # Deterministic per-(image, variant) RNG. String seeding
            # is SHA-512 based (stable across processes and runs);
            # cv2's RNG drives grabCut internals. Seeding per image
            # rather than once per run means resume point, shard
            # order, and skip patterns cannot shift anyone else's
            # transforms — same seed, same forgery, always.
            #
            # Each pass gets its own seed STREAM: retrying a failed
            # (image, variant) with the identical stream would just
            # replay the failed transform/destination draws. The
            # stream tag keeps retries deterministic while giving
            # them genuinely fresh draws.
            if extend_variants:
                stream = "ext2:" if reuse_sources else "ext1:"
            elif retry_failed:
                stream = "rec1:"
            else:
                stream = ""
            random.seed(f"{seed}:{stream}{img_id}:v{variant}")
            cv2.setRNGSeed(
                (seed * 1_000_003 + img_id * 31 + variant
                 + (7_777_777 if stream else 0)) % 2**31
            )

            banned = {
                c for c, n in cat_counts.items() if n >= cat_cap
            }
            try:
                result = generator.generate(
                    {"id": img_id, "file_name": file_name},
                    img_path,
                    anns,
                    coco,
                    stuff_coco=stuff_coco,
                    use_perspective_scale=cfg.use_perspective_scale,
                    use_supercategory_pool=cfg.use_supercategory_pool,
                    exclude_ann_ids=used_ann_ids,
                    banned_cats=banned,
                    out_suffix=f"_v{variant}" if variant else "",
                )
            except MemoryError:
                result = "ERROR: MemoryError"
            except Exception:
                result = "ERROR: " + traceback.format_exc(limit=3)

            # Track result
            if isinstance(result, dict):
                result["source_reused"] = int(
                    reuse_sources
                    and result.get("source_ann_id", -1)
                    in used_by_img.get(img_id, ())
                )
                successes.append(result)
                new_successes += 1
                ledger_f.write(json.dumps(result) + "\n")
                ledger_f.flush()
                used_ann_ids.add(result.get("source_ann_id", -1))
                cat = result.get("source_category_id", -1)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                logger.info(
                    "  [%d/%d] %s%s | tier=%s",
                    len(successes), cfg.num_images,
                    file_name,
                    f" v{variant}" if variant else "",
                    result.get("placement_tier", "?"),
                )
            else:
                failures.append(result)
                first_line = (
                    str(result).strip().splitlines() or ["unknown"]
                )[0][:200]
                skips_f.write(f"{img_id}\t{first_line}\n")
                skips_f.flush()
                # Image-level failures can't succeed on another
                # variant; object/placement-level ones might.
                if any(k in str(result) for k in (
                    "imread_failed", "low_resolution",
                    "grayscale_image", "no_suitable_annotations",
                )):
                    break

        gc.collect()

    ledger_f.close()
    skips_f.close()

    # ── Summary ──────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info(
        "Stage %s complete. Total: %d (new this run: %d) | "
        "Skipped/Failed this run: %d",
        cfg.name, len(successes), new_successes, len(failures),
    )

    if successes:
        df = pd.DataFrame(successes)
        meta_path = os.path.join(stage_output, "metadata.csv")
        df.to_csv(meta_path, index=False)
        logger.info("Metadata saved → %s", meta_path)

    # Provenance manifest — makes every run reproducible/citable.
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip() or "unknown"
    except Exception:
        git_commit = "unknown"
    manifest = {
        "stage": cfg.name,
        "seed": seed,
        "shard": shard,
        "num_shards": num_shards,
        "target_num_images": cfg.num_images,
        "generated": len(successes),
        "generated_this_run": new_successes,
        "failed_attempts": len(failures),
        "git_commit": git_commit,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "config": dataclasses.asdict(cfg),
    }
    # Per-stage manifest name so a recovery pass in the same output
    # dir never overwrites the strict run's provenance.
    manifest_name = f"manifest_{cfg.name}.json"
    with open(os.path.join(stage_output, manifest_name), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest → %s/%s", stage_output, manifest_name)

    # Write failure log
    fail_path = os.path.join(output_logs, "failed_samples.txt")
    with open(fail_path, "w") as f:
        f.write(f"Stage: {cfg.name}\n")
        f.write(f"Total Failed Attempts: {len(failures)}\n\n")
        # Count failure reasons (first line, capped length, so real
        # exception messages survive instead of being cut at ':')
        reason_counts = {}
        for line in failures:
            reason = str(line).strip().splitlines()[0][:100]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        f.write("Failure breakdown:\n")
        for reason, count in sorted(
            reason_counts.items(), key=lambda x: -x[1]
        ):
            f.write(f"  {count:4d} × {reason}\n")
        f.write("\nFirst 200 failures:\n")
        for line in failures[:200]:
            f.write(f"  {line}\n")

    logger.info("Failure log → %s", fail_path)
    return successes, stage_output


# ── Visualisation ────────────────────────────────────────────────
def visualize_stage(stage_output, max_samples=10, coco_root=None):
    """Visualise generated samples for a stage as a grid PNG."""
    from visualize import show_samples

    meta_path = os.path.join(stage_output, "metadata.csv")
    if not os.path.exists(meta_path):
        logger.warning("No metadata.csv found at %s", meta_path)
        return None

    return show_samples(
        meta_path=meta_path,
        coco_images_dir=coco_paths(coco_root)[0],
        output_dir=stage_output,
        max_samples=max_samples,
    )


# ── Main entry point ─────────────────────────────────────────────
def main():
    """CLI entry point for stage-aware dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="COCO-CMFD copy-move forgery dataset generator",
    )
    parser.add_argument(
        "--stage", default="stage1",
        choices=["stage1", "stage1_clean", "stage1_recovery",
                 "stage1_recovery_v2", "stage2", "stage2_synthetic"],
        help="Generation stage preset (default: stage1)",
    )
    parser.add_argument(
        "--num-images", type=int, default=None,
        help="Number of forged images to generate "
             "(default: stage preset value)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--visualize", type=int, default=0, metavar="N",
        help="Render a QA grid of N samples after generation",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the stage output directory "
             "(default: output/<stage_name>)",
    )
    parser.add_argument(
        "--shard", type=int, default=0,
        help="Shard index for full-COCO runs (default: 0)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total shards; >1 partitions all of COCO instead of "
             "sampling subset_size images (default: 1)",
    )
    parser.add_argument(
        "--subset-size", type=int, default=None,
        help="Override the preset's annotation-subset size "
             "(ignored when --num-shards > 1)",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Recovery mode: retry images that previously failed "
             "for recoverable reasons (use with --stage "
             "stage1_recovery)",
    )
    parser.add_argument(
        "--extend-variants", action="store_true",
        help="Extension mode: generate additional variants from "
             "already-successful images (unused source objects, "
             "continued variant indices)",
    )
    parser.add_argument(
        "--reuse-sources", action="store_true",
        help="With --extend-variants: allow proven source objects "
             "to be pasted again at new destinations (rows marked "
             "source_reused=1)",
    )
    parser.add_argument(
        "--coco-root", default=None,
        help="MS-COCO 2017 root containing train2017/ and "
             "annotations/ (default: $COCO_ROOT or ~/coco)",
    )
    args = parser.parse_args()

    successes, stage_output = run_stage(
        args.stage, num_images=args.num_images, seed=args.seed,
        output_dir=args.output_dir,
        shard=args.shard, num_shards=args.num_shards,
        subset_size=args.subset_size, retry_failed=args.retry_failed,
        extend_variants=args.extend_variants,
        reuse_sources=args.reuse_sources,
        coco_root=args.coco_root,
    )
    if args.visualize > 0:
        visualize_stage(
            stage_output, max_samples=args.visualize,
            coco_root=args.coco_root,
        )
    return successes


if __name__ == "__main__":
    main()