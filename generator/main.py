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
* Image downscale (configurable) before processing.
* gc.collect() after every image.
* resource.setrlimit as a safety net.
"""

import gc
import json
import logging
import os
import random
import resource
import subprocess
import sys

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
DEFAULT_COCO_PATH = "/home/harshita/coco/train2017"
DEFAULT_ANNOT_PATH = (
    "/home/harshita/coco/annotations/instances_train2017.json"
)
DEFAULT_STUFF_PATH = (
    "/home/harshita/coco/annotations/stuff_train2017.json"
)
DEFAULT_OUTPUT_DIR = "/home/harshita/projects/coco-cmfd/output"

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
                               n_images=200, seed=1337):
    """
    Create a small annotation subset using a subprocess.

    The subprocess loads the full 449 MB JSON in its own address
    space, filters it down, writes the result, and exits —
    returning ALL RAM to the OS.
    """
    if os.path.exists(subset_path):
        logger.info("Reusing cached subset: %s", subset_path)
        return

    logger.info(
        "Creating annotation subset (%d images) via subprocess...",
        n_images,
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


def _maybe_downscale(img_path, max_side):
    """
    Downscale an image if its longer side exceeds *max_side*.

    Returns the (possibly new) path.  Downscaling cuts numpy buffer
    sizes quadratically — the single biggest per-image RAM saving.
    """
    if max_side <= 0:
        return img_path

    img = cv2.imread(img_path)
    if img is None:
        return img_path

    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        del img
        return img_path

    scale = max_side / max(h, w)
    resized = cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    del img

    tmp_path = os.path.join(
        "/tmp", f"cmfd_{os.path.basename(img_path)}"
    )
    cv2.imwrite(tmp_path, resized)
    del resized
    return tmp_path


# ── Stage-aware generation ───────────────────────────────────────
def run_stage(stage_name, num_images=None, seed=SEED):
    """
    Run dataset generation for a given stage.

    Parameters
    ----------
    stage_name : str
        'stage1' or 'stage2'.
    num_images : int, optional
        Override the default number of images.
    seed : int
        Random seed for reproducibility.
    """
    cfg = get_stage_config(stage_name)
    if num_images is not None:
        cfg = get_stage_config(stage_name, num_images=num_images)

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

    # Output directories (stage-specific)
    stage_output = os.path.join(DEFAULT_OUTPUT_DIR, cfg.name)
    output_tampered = os.path.join(stage_output, "tampered")
    output_masks = os.path.join(stage_output, "masks")
    output_logs = os.path.join(stage_output, "logs")
    for d in (output_tampered, output_masks, output_logs):
        os.makedirs(d, exist_ok=True)

    # ── Step 1: annotation subset (subprocess) ──────────────────
    subset_path = os.path.join(
        stage_output, "annotations_subset.json"
    )
    _create_subset_annotations(
        DEFAULT_ANNOT_PATH, subset_path,
        n_images=cfg.subset_size, seed=seed,
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
        ids_path = os.path.join(stage_output, "image_ids.json")
        with open(ids_path, "w") as f:
            json.dump(img_ids, f)

        stuff_subset_path = os.path.join(
            stage_output, "stuff_subset.json"
        )
        has_stuff = _create_stuff_subset(
            DEFAULT_STUFF_PATH, stuff_subset_path, ids_path,
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

    successes = []
    failures = []

    for img_id in img_ids:
        if len(successes) >= cfg.num_images:
            break

        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        raw_path = os.path.join(DEFAULT_COCO_PATH, file_name)

        # Downscale if configured
        img_path = _maybe_downscale(raw_path, cfg.max_side)

        # Annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        try:
            result = generator.generate(
                {"id": img_id, "file_name": file_name},
                img_path,
                anns,
                coco,
                stuff_coco=stuff_coco,
                use_perspective_scale=cfg.use_perspective_scale,
                use_supercategory_pool=cfg.use_supercategory_pool,
            )
        except MemoryError:
            result = "ERROR: MemoryError — reduce max_side"
        except Exception as exc:
            result = f"ERROR: {exc}"

        # Track result
        if isinstance(result, dict):
            successes.append(result)
            logger.info(
                "  [%d/%d] %s | tier=%s",
                len(successes), cfg.num_images,
                file_name,
                result.get("placement_tier", "?"),
            )
        else:
            failures.append(result)

        # Clean up temp file and encourage GC
        if img_path != raw_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info(
        "Stage %s complete. Generated: %d | Skipped/Failed: %d",
        cfg.name, len(successes), len(failures),
    )

    if successes:
        df = pd.DataFrame(successes)
        meta_path = os.path.join(stage_output, "metadata.csv")
        df.to_csv(meta_path, index=False)
        logger.info("Metadata saved → %s", meta_path)

    # Write failure log
    fail_path = os.path.join(output_logs, "failed_samples.txt")
    with open(fail_path, "w") as f:
        f.write(f"Stage: {cfg.name}\n")
        f.write(f"Total Failed Attempts: {len(failures)}\n\n")
        # Count failure reasons
        reason_counts = {}
        for line in failures:
            reason = str(line).split(":")[0] + ":" + str(line).split(":")[1] if ":" in str(line) else str(line)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        f.write("Failure breakdown:\n")
        for reason, count in sorted(
            reason_counts.items(), key=lambda x: -x[1]
        ):
            f.write(f"  {count:4d} × {reason}\n")
        f.write(f"\nFirst 200 failures:\n")
        for line in failures[:200]:
            f.write(f"  {line}\n")

    logger.info("Failure log → %s", fail_path)
    return successes, stage_output


# ── Visualisation ────────────────────────────────────────────────
def visualize_stage(stage_output, max_samples=10):
    """Visualise generated samples for a stage as a grid PNG."""
    from visualize import show_samples

    meta_path = os.path.join(stage_output, "metadata.csv")
    if not os.path.exists(meta_path):
        logger.warning("No metadata.csv found at %s", meta_path)
        return None

    return show_samples(
        meta_path=meta_path,
        coco_images_dir=DEFAULT_COCO_PATH,
        output_dir=stage_output,
        max_samples=max_samples,
    )


# ── Example: run both stages ────────────────────────────────────
def example_stage1_generation(num_images=10):
    """Generate Stage 1 (clean) samples and visualise."""
    successes, stage_output = run_stage(
        "stage1", num_images=num_images
    )
    visualize_stage(stage_output, max_samples=num_images)
    return successes


def example_stage2_generation(num_images=10):
    """Generate Stage 2 (synthetic) samples and visualise."""
    successes, stage_output = run_stage(
        "stage2", num_images=num_images
    )
    visualize_stage(stage_output, max_samples=num_images)
    return successes


# ── Main entry point ─────────────────────────────────────────────
def main():
    """Entry point: generate samples for both stages."""
    logger.info("COCO-CMFD Stage-Aware Pipeline")
    logger.info("Generating 10 samples per stage for sanity check")

    example_stage1_generation(num_images=10)
    example_stage2_generation(num_images=10)


if __name__ == "__main__":
    main()