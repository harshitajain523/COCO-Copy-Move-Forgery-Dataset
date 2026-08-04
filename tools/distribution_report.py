"""
Quality gate 2: distribution report.

Summarises category balance, transform parameters, placement tiers,
and paste sizes for a generated stage, so the dataset isn't
accidentally 80% "person" or all tiny patches. Writes a text report
and a histogram PNG next to metadata.csv.

Usage:
    python tools/distribution_report.py output/stage1_full
"""

import argparse
import ast
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def report(stage_dir):
    df = pd.read_csv(os.path.join(stage_dir, "metadata.csv"))
    lines = [f"Distribution report — {stage_dir}",
             f"Samples: {len(df)}", ""]

    # Paste size as fraction of image area (from target bbox)
    tgt = df["target_bbox"].apply(ast.literal_eval)
    area_frac = tgt.apply(lambda b: b[2] * b[3]) / (
        df["image_height"] * df["image_width"]
    )

    lines.append("Top categories (share):")
    counts = df["source_category_name"].value_counts()
    for name, n in counts.head(15).items():
        lines.append(f"  {n:6d}  {n / len(df):6.1%}  {name}")
    lines.append(f"  ({len(counts)} distinct categories; "
                 f"max share {counts.iloc[0] / len(df):.1%})")
    lines.append("")

    for col in ("source_supercategory", "placement_tier",
                "horizon_tier"):
        lines.append(f"{col}:")
        for name, n in df[col].value_counts().items():
            lines.append(f"  {n:6d}  {n / len(df):6.1%}  {name}")
        lines.append("")

    lines.append(f"flip rate: {df['flip'].mean():.1%}")
    lines.append(f"scale_factor: min={df['scale_factor'].min():.3f} "
                 f"mean={df['scale_factor'].mean():.3f} "
                 f"max={df['scale_factor'].max():.3f}")
    lines.append(f"rotation_angle: min={df['rotation_angle'].min():.1f} "
                 f"mean={df['rotation_angle'].mean():.1f} "
                 f"max={df['rotation_angle'].max():.1f}")
    lines.append(f"paste bbox area fraction: "
                 f"min={area_frac.min():.4f} "
                 f"median={area_frac.median():.4f} "
                 f"max={area_frac.max():.4f}")

    txt_path = os.path.join(stage_dir, "distribution_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved → {txt_path}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].hist(df["scale_factor"], bins=30)
    axs[0, 0].set_title("scale_factor")
    axs[0, 1].hist(df["rotation_angle"], bins=30)
    axs[0, 1].set_title("rotation_angle (deg)")
    axs[1, 0].hist(area_frac, bins=30)
    axs[1, 0].set_title("paste bbox area fraction")
    counts.head(20)[::-1].plot.barh(ax=axs[1, 1], fontsize=7)
    axs[1, 1].set_title("top-20 categories")
    fig.suptitle(f"COCO-CMFD distribution — {os.path.basename(stage_dir)}")
    fig.tight_layout()
    png_path = os.path.join(stage_dir, "distribution_report.png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print(f"Saved → {png_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage_dir")
    a = p.parse_args()
    report(a.stage_dir)


if __name__ == "__main__":
    main()
