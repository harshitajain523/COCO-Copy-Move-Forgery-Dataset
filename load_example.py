"""Minimal loader for the COCO-CMFD dataset from the Hugging Face Hub."""

import numpy as np
from datasets import load_dataset

REPO = "harshitajainn/coco-cmfd"  

ds = load_dataset(REPO, split="train")            # add streaming=True for no local copy
strict = ds.filter(lambda r: r["gate_profile"] == "strict")

s = ds[0]
image = s["image"]                                 # PIL.Image, forged
trimap = np.array(s["trimap"])                     # 0 bg / 128 source / 255 target
source_mask = trimap == 128
target_mask = trimap == 255
hard_negatives = np.array(s["hard_negatives"]) > 0  # other objects in the scene
patches = np.array(s["patch_labels"], dtype=np.int8).reshape(
    s["patch_labels_shape"]                        # -1 ignore / 0 bg / 1 source / 2 target
)

print(f"{len(ds)} samples ({len(strict)} strict) | {image.size} "
      f"| {s['source_category_name']} | patches {patches.shape}")
