"""
Quality gate 3: visual contact sheet.

Per sample row: original | forged | trimap overlay | zoomed paste
boundary. Saved to samples/ for eyeball review before any scale-up.

Usage:
    python tools/contact_sheet.py output/stage1_full \
        [--coco-dir ~/coco/train2017] [-n 20] [--seed 0]
"""

import argparse
import ast
import datetime
import os


def _default_coco_dir():
    """COCO train2017 dir from $COCO_ROOT (default ~/coco)."""
    return os.path.join(
        os.environ.get("COCO_ROOT", os.path.expanduser("~/coco")),
        "train2017",
    )

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _overlay(forged_bgr, tri):
    """Source region green, target region red, over the forged image."""
    out = forged_bgr.copy()
    out[tri == 128] = (0.4 * out[tri == 128]
                       + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    out[tri == 255] = (0.4 * out[tri == 255]
                       + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    return out


def sheet(stage_dir, coco_dir, n, seed):
    df = pd.read_csv(os.path.join(stage_dir, "metadata.csv"))
    df = df.sample(n=min(n, len(df)), random_state=seed) \
           .reset_index(drop=True)

    fig, axs = plt.subplots(len(df), 4, figsize=(18, 4 * len(df)))
    if len(df) == 1:
        axs = axs[None, :]

    for i, row in df.iterrows():
        forged = cv2.imread(
            os.path.join(stage_dir, "tampered", row["image_filename"])
        )
        tri = cv2.imread(
            os.path.join(stage_dir, "masks", row["mask_filename"]),
            cv2.IMREAD_GRAYSCALE,
        )
        stem = os.path.splitext(row["image_filename"])[0]
        orig = cv2.imread(
            os.path.join(coco_dir, stem.split("_v")[0] + ".jpg")
        )
        if forged is None or tri is None:
            for j in range(4):
                axs[i, j].axis("off")
            continue

        title = (f"{row['image_filename']} | "
                 f"{row['source_category_name']} | "
                 f"{row.get('placement_tier', '?')}/"
                 f"{row.get('horizon_tier', '?')} | "
                 f"s={row['scale_factor']} r={row['rotation_angle']}")

        axs[i, 0].imshow(_rgb(orig) if orig is not None else [[0]])
        axs[i, 0].set_title(f"original — {title}", fontsize=8)
        axs[i, 1].imshow(_rgb(forged))
        axs[i, 1].set_title("forged", fontsize=8)
        axs[i, 2].imshow(_rgb(_overlay(forged, tri)))
        axs[i, 2].set_title("overlay (src=green tgt=red)", fontsize=8)

        # Zoomed paste boundary: target bbox + 25% margin
        x, y, w, h = ast.literal_eval(row["target_bbox"])
        mx, my = int(w * 0.25) + 8, int(h * 0.25) + 8
        H, W = forged.shape[:2]
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(W, x + w + mx), min(H, y + h + my)
        crop = forged[y1:y2, x1:x2].copy()
        # Draw the target contour on the zoom for boundary inspection
        tgt = ((tri[y1:y2, x1:x2] == 255) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            tgt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(crop, contours, -1, (0, 255, 255), 1)
        axs[i, 3].imshow(_rgb(crop))
        axs[i, 3].set_title("paste boundary (zoom)", fontsize=8)

        for j in range(4):
            axs[i, j].axis("off")

    fig.tight_layout()
    os.makedirs("samples", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(
        "samples",
        f"contact_sheet_{os.path.basename(stage_dir)}_{stamp}.png",
    )
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Contact sheet → {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage_dir")
    p.add_argument("--coco-dir", default=_default_coco_dir())
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sheet(a.stage_dir, a.coco_dir, a.n, a.seed)


if __name__ == "__main__":
    main()
