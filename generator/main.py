"""
Entry point for COCO-CMFD dataset generation pipeline.

STABLE WSL VERSION  (8 GB RAM safe)
====================================
Root cause of OOM: the 449 MB ``instances_train2017.json`` expands to
~2 GB of Python objects inside pycocotools.  This version avoids that
by running annotation filtering in a SUBPROCESS (isolated memory) and
only loading a tiny (~200 KB) subset file in the main process.

Key design decisions
--------------------
* Annotation pre-filter via subprocess  →  main process never loads
  the 449 MB file.
* Single-process only — multiprocessing is removed entirely.
* Image downscale (``max_side=800``) before processing.
* ``gc.collect()`` after every image.
* ``resource.setrlimit`` as a safety net.
"""

import gc
import os
import random
import resource
import subprocess
import sys

import cv2
import pandas as pd
from pycocotools.coco import COCO

from generator import CopyMoveGenerator

# ── Default paths ────────────────────────────────────────────────
DEFAULT_COCO_PATH = "/home/harshita/coco/train2017"
DEFAULT_ANNOT_PATH = (
    "/home/harshita/coco/annotations/instances_train2017.json"
)
DEFAULT_OUTPUT_DIR = "/home/harshita/projects/coco-cmfd/output"

# ── Tunables ─────────────────────────────────────────────────────
NUM_IMAGES = 5          # Forgeries to generate
MAX_SIDE = 800          # Downscale longer side (saves RAM hugely)
SUBSET_SIZE = 200       # Images in the annotation subset
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
        print(
            f"[mem] RSS limit: "
            f"{soft_gb:.1f} GB soft / {hard_gb:.1f} GB hard"
        )
    except Exception as exc:
        print(f"[mem] Could not set limit (non-fatal): {exc}")


def _create_subset_annotations(annot_path, subset_path,
                               n_images=200, seed=1337):
    """
    Create a small annotation subset using a subprocess.

    The subprocess loads the full 449 MB JSON in its own address space,
    filters it down, writes the result, and exits — returning ALL RAM
    to the OS.  The main process only ever loads the tiny output file.
    """
    if os.path.exists(subset_path):
        print(f"[subset] Reusing cached subset: {subset_path}")
        return

    print(f"[subset] Creating annotation subset ({n_images} images)…")
    print("[subset] This runs in a subprocess (may take 30-90 s)…")

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
        timeout=600,              # generous timeout
    )

    if result.returncode != 0:
        print(
            f"[subset] FAILED:\n{result.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Relay subprocess stdout
    for line in result.stdout.strip().splitlines():
        print(f"[subset] {line}")


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

    tmp_path = os.path.join("/tmp", f"cmfd_{os.path.basename(img_path)}")
    cv2.imwrite(tmp_path, resized)
    del resized
    return tmp_path


# ── Example: generate dataset ───────────────────────────────────
def example_dataset_generation():
    """Generate copy-move forgery samples using COCO train2017."""
    # Safety first
    _apply_memory_limit(MEM_LIMIT_SOFT_GB, MEM_LIMIT_HARD_GB)
    cv2.setNumThreads(1)

    # Output directories
    output_tampered = os.path.join(DEFAULT_OUTPUT_DIR, "tampered")
    output_masks = os.path.join(DEFAULT_OUTPUT_DIR, "binary_masks")
    output_logs = os.path.join(DEFAULT_OUTPUT_DIR, "logs")
    for d in (output_tampered, output_masks, output_logs):
        os.makedirs(d, exist_ok=True)

    # ── Step 1: create a tiny annotation subset (subprocess) ─────
    subset_path = os.path.join(DEFAULT_OUTPUT_DIR, "annotations_subset.json")
    _create_subset_annotations(
        DEFAULT_ANNOT_PATH, subset_path,
        n_images=SUBSET_SIZE, seed=SEED,
    )

    # ── Step 2: load the tiny subset (~200 KB, instant) ──────────
    print("\nLoading annotation subset …")
    coco = COCO(subset_path)
    img_ids = coco.getImgIds()
    print(f"Subset contains {len(img_ids)} images\n")

    # ── Step 3: create generator with sensible defaults ──────────
    generator = CopyMoveGenerator(
        output_dir_tampered=output_tampered,
        output_dir_masks=output_masks,
    )

    # ── Step 4: generate forgeries ───────────────────────────────
    print(f"Generating {NUM_IMAGES} copy-move forgeries …\n")

    rng = random.Random(SEED)
    rng.shuffle(img_ids)

    successes = []
    failures = []

    for img_id in img_ids:
        if len(successes) >= NUM_IMAGES:
            break

        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        raw_path = os.path.join(DEFAULT_COCO_PATH, file_name)

        # Downscale to save RAM
        img_path = _maybe_downscale(raw_path, MAX_SIDE)

        # Annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        try:
            result = generator.generate(
                {"id": img_id, "file_name": file_name},
                img_path, anns, coco,
            )
        except MemoryError:
            result = "ERROR: MemoryError — reduce MAX_SIDE"
        except Exception as exc:
            result = f"ERROR: {exc}"

        # Track result
        if isinstance(result, dict):
            successes.append(result)
            print(f"  ✓ [{len(successes)}/{NUM_IMAGES}] {file_name}")
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
    print(f"\n{'=' * 50}")
    print(
        f"Done.  Generated: {len(successes)}  |  "
        f"Skipped/Failed: {len(failures)}"
    )

    if successes:
        df = pd.DataFrame(successes)
        meta_path = os.path.join(DEFAULT_OUTPUT_DIR, "metadata.csv")
        df.to_csv(meta_path, index=False)
        print(f"Metadata saved → {meta_path}")

    # Write failure log
    with open(os.path.join(output_logs, "failed_samples.txt"), "w") as f:
        f.write(f"Total Failed Attempts: {len(failures)}\n\n")
        for line in failures[:200]:
            f.write(f"{line}\n")

    return successes


# ── Example: visualise results ───────────────────────────────────
def example_visualize():
    """Visualise the generated samples as a grid saved to PNG."""
    from visualize import show_samples

    meta_path = os.path.join(DEFAULT_OUTPUT_DIR, "metadata.csv")

    if not os.path.exists(meta_path):
        print("No metadata.csv found — run generation first.")
        return None

    return show_samples(
        meta_path=meta_path,
        coco_images_dir=DEFAULT_COCO_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        max_samples=5,
    )


# ── Main entry point ─────────────────────────────────────────────
def main():
    """Entry point: generate 5 samples, then visualise."""
    example_dataset_generation()
    example_visualize()


if __name__ == "__main__":
    main()