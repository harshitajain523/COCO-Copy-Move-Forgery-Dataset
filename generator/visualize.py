"""
Visualization module for COCO-CMFD generated samples.

Displays original images alongside tampered versions, binary masks,
and tri-maps in a matplotlib grid for visual inspection.  Saves the
result as a PNG file (WSL-safe, no GUI needed).

Supports both old flat output layout and new stage-specific layout.
"""

import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Use non-interactive backend (safe for WSL / headless)
matplotlib.use("Agg")


def _load_rgb(path):
    """Load an image file and convert BGR → RGB for matplotlib."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show_samples(meta_path, coco_images_dir, output_dir,
                 max_samples=10):
    """
    Display generated samples as a grid and save to PNG.

    Layout per row: Original | Tampered | Tri-Map | Binary Mask

    Parameters
    ----------
    meta_path : str
        Path to metadata.csv produced by the generator.
    coco_images_dir : str
        Path to the original COCO images directory.
    output_dir : str
        Path to the stage output directory.
    max_samples : int
        Maximum number of rows to display.

    Returns
    -------
    str or None
        Path to the saved PNG, or None if nothing to show.
    """
    df = pd.read_csv(meta_path)
    df = df.head(max_samples).reset_index(drop=True)
    n = len(df)

    if n == 0:
        print("No samples to visualise.")
        return None

    tampered_dir = os.path.join(output_dir, "tampered")
    mask_dir = os.path.join(output_dir, "masks")

    # Check if binary_masks_flat directory exists (new layout)
    binary_dir = os.path.join(output_dir, "binary_masks_flat")
    has_binary = os.path.isdir(binary_dir)

    # Determine number of columns
    n_cols = 4 if has_binary else 3

    fig, axs = plt.subplots(n, n_cols, figsize=(5 * n_cols, 4 * n))

    # Handle single-row case where axs is 1-D
    if n == 1:
        axs = axs[None, :]

    for i, row in df.iterrows():
        # Resolve paths
        tampered_path = os.path.join(
            tampered_dir, row["image_filename"]
        )
        orig_name = row["image_filename"].replace(".png", ".jpg")
        orig_path = os.path.join(coco_images_dir, orig_name)
        mask_path = os.path.join(mask_dir, row["mask_filename"])

        # Load images
        tampered = _load_rgb(tampered_path)
        original = _load_rgb(orig_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Build row title with metadata
        tier = row.get("placement_tier", "?")
        cat = row.get("source_category_name", "?")
        scale = row.get("scale_factor", "?")
        rot = row.get("rotation_angle", "?")
        title_suffix = (
            f" | {cat} | tier={tier} | s={scale} r={rot}"
        )

        # Column 0: Original
        axs[i, 0].imshow(
            original if original is not None else [[0]]
        )
        axs[i, 0].set_title(
            f"#{i + 1} Original{title_suffix}", fontsize=9
        )
        axs[i, 0].axis("off")

        # Column 1: Tampered
        axs[i, 1].imshow(
            tampered if tampered is not None else [[0]]
        )
        axs[i, 1].set_title("Tampered", fontsize=10)
        axs[i, 1].axis("off")

        # Column 2: Tri-Map
        axs[i, 2].imshow(
            mask if mask is not None else [[0]], cmap="gray"
        )
        axs[i, 2].set_title(
            "Tri-Map (128=src, 255=tgt)", fontsize=9
        )
        axs[i, 2].axis("off")

        # Column 3: Binary mask (if available)
        if has_binary:
            bin_name = row.get("binary_mask_filename", "")
            if bin_name:
                bin_path = os.path.join(binary_dir, bin_name)
                bin_mask = cv2.imread(
                    bin_path, cv2.IMREAD_GRAYSCALE
                )
            else:
                bin_mask = None
            axs[i, 3].imshow(
                bin_mask if bin_mask is not None else [[0]],
                cmap="gray",
            )
            axs[i, 3].set_title("Binary Mask", fontsize=10)
            axs[i, 3].axis("off")

    # Stage name from directory
    stage_name = os.path.basename(output_dir)
    plt.suptitle(
        f"COCO-CMFD — {stage_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save to file
    save_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualisation saved → {save_path}")
    return save_path


if __name__ == "__main__":
    # Standalone usage (default paths)
    out = "/home/harshita/projects/coco-cmfd/output"
    coco = "/home/harshita/coco/train2017"

    # Try stage-specific directories first
    for stage in ["stage1_clean", "stage2_synthetic"]:
        stage_dir = os.path.join(out, stage)
        meta = os.path.join(stage_dir, "metadata.csv")
        if os.path.exists(meta):
            print(f"\n--- {stage} ---")
            show_samples(meta, coco, stage_dir, max_samples=10)

    # Fallback to flat layout
    meta = os.path.join(out, "metadata.csv")
    if os.path.exists(meta):
        print("\n--- legacy flat layout ---")
        show_samples(meta, coco, out, max_samples=5)
