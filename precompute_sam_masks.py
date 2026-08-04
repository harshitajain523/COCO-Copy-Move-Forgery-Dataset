"""
Precompute SAM masks for COCO annotations.

Uses MobileSAM (ViT-Tiny) to generate high-fidelity binary masks
from COCO bounding-box prompts.  Masks are saved as PNG files and
consumed by the generator to replace coarse COCO polygon masks.

Design note:
    Masks are saved to output/sam_masks/<img_id>_<ann_id>.png.
    The generator resolves this path relative to the project root
    using the SAM_MASKS_ROOT constant so that runs from any working
    directory resolve correctly.

Usage:
    python precompute_sam_masks.py
"""

import json
import os

import cv2
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry
from pycocotools.coco import COCO
from tqdm import tqdm


# ── Project-root-relative SAM mask directory ─────────────────────
# This must match generator.py's SAM_MASKS_ROOT constant so that
# the generator can reliably find pre-computed masks.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_MASKS_ROOT = os.path.join(_SCRIPT_DIR, "output", "sam_masks")


def _download_checkpoint_if_needed(checkpoint_path):
    """
    Download MobileSAM weights if the checkpoint is missing.

    Parameters
    ----------
    checkpoint_path : str
        Local path where the checkpoint should exist.
    """
    if os.path.exists(checkpoint_path):
        return
    import urllib.request
    url = (
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/"
        "weights/mobile_sam.pt"
    )
    print(f"Downloading MobileSAM weights from {url} ...")
    urllib.request.urlretrieve(url, checkpoint_path)
    print(f"Saved to {checkpoint_path}")


def build_sam_predictor(checkpoint_path):
    """
    Instantiate and return a MobileSAM SamPredictor on best device.

    Parameters
    ----------
    checkpoint_path : str
        Path to mobile_sam.pt checkpoint.

    Returns
    -------
    SamPredictor
        Loaded, eval-mode predictor ready for set_image().
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _download_checkpoint_if_needed(checkpoint_path)

    model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


def run_precompute(
    ann_file,
    img_dir,
    out_dir,
    checkpoint_path="mobile_sam.pt",
):
    """
    Iterate over COCO annotations and save SAM-predicted binary masks.

    Skips masks that already exist on disk (resume-safe).

    Parameters
    ----------
    ann_file : str
        Path to COCO-format annotations JSON (subset or full).
    img_dir : str
        Directory containing the raw COCO JPEG images.
    out_dir : str
        Directory where masks will be saved as PNG files.
    checkpoint_path : str
        Path to mobile_sam.pt weights file.
    """
    os.makedirs(out_dir, exist_ok=True)

    predictor = build_sam_predictor(checkpoint_path)
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    n_saved = 0
    n_skipped = 0

    for img_id in tqdm(img_ids, desc="Images"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            continue

        # Load image; SAM requires RGB
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image once per image (amortises encoder cost)
        try:
            predictor.set_image(image_rgb)
        except Exception as exc:
            print(f"  [WARN] set_image failed for {img_id}: {exc}")
            continue

        anns = coco.loadAnns(
            coco.getAnnIds(imgIds=img_id, iscrowd=False)
        )
        for ann in anns:
            ann_id = ann["id"]
            out_path = os.path.join(
                out_dir, f"{img_id}_{ann_id}.png"
            )

            # Skip already-computed masks (resume-safe)
            if os.path.exists(out_path):
                n_skipped += 1
                continue

            x, y, w, h = ann["bbox"]
            input_box = np.array([x, y, x + w, y + h])

            try:
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                best_mask = (masks[0] * 255).astype(np.uint8)
                cv2.imwrite(out_path, best_mask)
                n_saved += 1
            except Exception as exc:
                print(
                    f"  [WARN] SAM predict failed for "
                    f"img={img_id} ann={ann_id}: {exc}"
                )

    print(
        f"\nDone. Saved: {n_saved}  |  Skipped (cached): {n_skipped}"
    )


def main():
    """Entry point for pre-computing SAM masks."""
    # Default to the stage1 annotations subset; fall back to stage2.
    project_root = os.path.dirname(os.path.abspath(__file__))

    for stage in ("stage1_clean", "stage2_synthetic"):
        candidate = os.path.join(
            project_root, "output", stage, "annotations_subset.json"
        )
        if os.path.exists(candidate):
            ann_file = candidate
            break
    else:
        raise FileNotFoundError(
            "No annotations_subset.json found. "
            "Run the generator first to create a subset."
        )

    img_dir = os.path.join(
        os.environ.get("COCO_ROOT", os.path.expanduser("~/coco")),
        "train2017",
    )
    checkpoint = os.environ.get(
        "MOBILE_SAM_CKPT", os.path.join(project_root, "mobile_sam.pt")
    )

    run_precompute(
        ann_file=ann_file,
        img_dir=img_dir,
        out_dir=SAM_MASKS_ROOT,
        checkpoint_path=checkpoint,
    )


if __name__ == "__main__":
    main()
