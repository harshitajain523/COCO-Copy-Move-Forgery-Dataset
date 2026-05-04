"""
Standalone script to filter COCO Stuff annotations into a small subset.

Mirrors the design of filter_annotations.py — runs as a SUBPROCESS
so the large stuff JSON (~800 MB → ~3 GB in memory) is loaded in an
isolated process. When this process exits, ALL memory is returned.

Usage:
    python filter_stuff.py <input> <output> <image_ids_json>

Where <image_ids_json> is a JSON file containing a list of image IDs
to keep (produced by filter_annotations.py or main.py).
"""

import json
import os
import sys


def filter_stuff_annotations(input_path, output_path, image_ids_path):
    """
    Load full COCO Stuff JSON, keep only annotations for given image IDs.

    Parameters
    ----------
    input_path : str
        Path to stuff_train2017.json (~800 MB).
    output_path : str
        Path to write the filtered subset.
    image_ids_path : str
        Path to JSON file with list of image IDs to keep.
    """
    # Load the target image IDs (tiny file)
    print(f"Loading image IDs from {image_ids_path} ...")
    with open(image_ids_path, "r") as f:
        image_ids = set(json.load(f))
    print(f"Keeping annotations for {len(image_ids)} images")

    # Load full stuff annotations
    print(f"Loading {input_path} (this may take 30-60 s) ...")
    with open(input_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print(
        f"Full stuff dataset: {len(images)} images, "
        f"{len(annotations)} annotations"
    )

    # Filter to chosen images
    chosen_images = [
        img for img in images if img["id"] in image_ids
    ]
    chosen_annotations = [
        ann for ann in annotations if ann["image_id"] in image_ids
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
        f"Stuff subset: {len(chosen_images)} images, "
        f"{len(chosen_annotations)} annotations ({size_kb:.0f} KB)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} "
            "<stuff_json> <output_json> <image_ids_json>"
        )
        sys.exit(1)

    filter_stuff_annotations(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        image_ids_path=sys.argv[3],
    )
