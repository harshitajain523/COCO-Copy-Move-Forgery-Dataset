"""
Standalone script to filter COCO annotation JSON into a small subset.

Designed to run as a SUBPROCESS so the large JSON (~449MB → ~2GB in
memory) is loaded in an isolated process.  When this process exits,
ALL memory is returned to the OS — the main generator process never
touches the 449MB file.

Usage:
    python filter_annotations.py <input> <output> <n_images> <seed>
"""

import json
import os
import random
import sys


def filter_annotations(input_path, output_path, n_images, seed):
    """Load full COCO JSON, extract a random subset, write to output."""
    print(f"Loading {input_path} ...")
    with open(input_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data.get("categories", [])

    print(
        f"Full dataset: {len(images)} images, "
        f"{len(annotations)} annotations"
    )

    # Pick random subset of images
    rng = random.Random(seed)
    rng.shuffle(images)
    chosen_images = images[:n_images]
    chosen_ids = {img["id"] for img in chosen_images}

    # Filter annotations to only include chosen images
    chosen_annotations = [
        ann for ann in annotations
        if ann["image_id"] in chosen_ids
    ]

    # Write compact subset
    subset = {
        "images": chosen_images,
        "annotations": chosen_annotations,
        "categories": categories,
    }

    with open(output_path, "w") as f:
        json.dump(subset, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(
        f"Subset: {len(chosen_images)} images, "
        f"{len(chosen_annotations)} annotations ({size_kb:.0f} KB)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} "
            "<input_json> <output_json> <n_images> <seed>"
        )
        sys.exit(1)

    filter_annotations(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        n_images=int(sys.argv[3]),
        seed=int(sys.argv[4]),
    )
