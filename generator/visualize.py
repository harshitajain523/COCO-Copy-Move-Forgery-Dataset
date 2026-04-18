"""
Visualization module for COCO-CMFD generated samples.

Displays original images alongside tampered versions and binary
masks in a matplotlib grid for visual inspection.  Saves the
result as a PNG file (WSL-safe, no GUI needed).
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


def show_samples(meta_path, coco_images_dir, output_dir, max_samples=5):
    """
    Display generated samples as a grid: Original / Tampered / Mask.

    Saves the visualisation to ``output_dir/visualization.png``.

    Parameters
    ----------
    meta_path : str
        Path to metadata.csv produced by the generator.
    coco_images_dir : str
        Path to the original COCO images directory.
    output_dir : str
        Path to the generator output directory.
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
    mask_dir = os.path.join(output_dir, "binary_masks")

    fig, axs = plt.subplots(n, 3, figsize=(15, 4 * n))

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

        # Plot row
        axs[i, 0].imshow(original if original is not None else [[0]])
        axs[i, 0].set_title(f"#{i + 1} Original", fontsize=12)
        axs[i, 0].axis("off")

        axs[i, 1].imshow(tampered if tampered is not None else [[0]])
        axs[i, 1].set_title("Tampered", fontsize=12)
        axs[i, 1].axis("off")

        axs[i, 2].imshow(
            mask if mask is not None else [[0]], cmap="gray"
        )
        axs[i, 2].set_title("Tri-Map Mask", fontsize=12)
        axs[i, 2].axis("off")

    plt.suptitle(
        "COCO-CMFD Generated Samples",
        fontsize=16,
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
    # Standalone usage
    out = "/home/harshita/projects/coco-cmfd/output"
    coco = "/home/harshita/coco/train2017"
    meta = os.path.join(out, "metadata.csv")

    if os.path.exists(meta):
        show_samples(meta, coco, out, max_samples=5)
    else:
        print("No metadata.csv found. Generate samples first.")
