---
license: cc-by-4.0
task_categories:
- image-segmentation
- image-classification
language:
- en
tags:
- image-forensics
- copy-move-forgery
- forgery-detection
- tamper-detection
- synthetic
- coco
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# COCO-CMFD

A synthetic copy-move forgery dataset generated from MS-COCO 2017, with
source/target-separated ground truth for copy-move forgery detection
(CMFD).

Each sample takes one annotated COCO object, applies a mild affine
transform, and pastes it elsewhere in the *same* image at a location
that passes scene-plausibility checks (support surface, horizon band,
perspective scale, occlusion). Ground truth is provided as a 3-class
trimap, a binary mask, a 16 px patch-label grid, and a hard-negative
mask of the scene's other objects.

Code, generation pipeline and quality tooling:
https://github.com/harshitajain523/COCO-Copy-Move-Forgery-Dataset

## Dataset details

| | |
|---|---|
| Samples | 12,271 |
| Source images | 7,150 distinct COCO train2017 images (1–8 samples each, mean 1.7) |
| Categories | 64 COCO categories across 9 supercategories |
| Resolution | native COCO, 320–640 px per side |
| Pasted region size | 2.5%–47.6% of image area (median 8.6%) |
| Transforms | scale 0.70–1.45, rotation ±10°, horizontal flip on 25.1% |
| Post-processing | none (no JPEG, noise or blur applied) |

### Fields

| Field | Type | Description |
|---|---|---|
| `image` | Image | Forged image (PNG, lossless) |
| `trimap` | Image | `0` background, `128` source, `255` target |
| `binary_mask` | Image | `0`/`255`, source ∪ target |
| `hard_negatives` | Image | `0`/`255`, all non-source annotated objects |
| `patch_labels` | list[int8] | Flattened 16 px grid: `-1` ignore, `0` bg, `1` source, `2` target |
| `patch_labels_shape` | list[int] | `[H/16, W/16]` for reshaping `patch_labels` |
| `gate_profile` | string | `strict`, `relaxed_v1`, or `relaxed_v2` (see below) |
| `source_reused` | int | `1` if this source object is also the source of another sample from the same image |
| `source_category_name` | string | COCO category of the copied object |
| `scale_factor`, `rotation_angle`, `flip` | float/int | Transform applied to the copy |
| `source_bbox`, `target_bbox` | string | `(x, y, w, h)` tuples |
| `placement_tier`, `horizon_tier` | string | Which candidate pool / vertical constraint accepted the placement |

24 metadata fields in total; `image_id` refers to the originating COCO
image.

Patch labels use a strict purity rule: a patch is labelled only if all
256 pixels share one class, otherwise `-1`. This keeps the copy-move
boundary out of patch-level contrastive objectives.

### Generation profiles

| `gate_profile` | Samples | Source-object gates |
|---|---|---|
| `strict` | 4,420 | edge margin 15 px, isolation 5×5, bbox fill ≥ 0.25, paste ≥ 2% of image |
| `relaxed_v1` | 166 | edge 8 px, isolation 3×3, fill ≥ 0.20 |
| `relaxed_v2` | 7,685 | as `relaxed_v1`, plus paste ≥ 1.2% of image |

Scene-plausibility gates are identical across all three profiles; only
source-object selectivity differs. Filter on `gate_profile == "strict"`
for the most conservative subset.

## Usage

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("harshitajainn/coco-cmfd", split="train")
s = ds[0]

image = s["image"]                              # PIL.Image
trimap = np.array(s["trimap"])
source_mask, target_mask = trimap == 128, trimap == 255
patches = np.array(s["patch_labels"], dtype=np.int8).reshape(
    s["patch_labels_shape"]
)
```

WebDataset tar shards are also published under `wds/` for
streaming-heavy pipelines; each sample key carries `.png`,
`.trimap.png`, `.binary.png`, `.hardneg.npy`, `.patches.npy` and
`.json`.

## Verification

Run on the released set with the tooling in the GitHub repository:

- **Mask alignment** (60 samples): IoU between changed pixels and the
  labelled target region — mean 1.0000, min 0.9997, with 0 pixels
  modified outside any mask. Blending uses an inward-only 2 px feather,
  so the alpha ramp stays inside the labelled region and the ground
  truth is exact rather than approximate.
- **Integrity**: 61,355/61,355 referenced files present, 400 sampled
  pixel checks pass.
- **Post-hoc plausibility audit**: 12,190 clean / 81 rejected (0.66%).

## Intended use and limitations

Intended for pretraining and evaluating copy-move forgery detectors,
including source/target discrimination and patch-level contrastive
objectives.

Limitations:

- **Synthetic**. Forgeries are algorithmically composited, not made by
  a human with intent to deceive. Detectors trained only on this data
  should be fine-tuned and evaluated on real-world benchmarks.
- **No post-processing**. Images are lossless PNG with no JPEG
  recompression, noise or blur. Robustness to those degradations is not
  exercised by this set.
- **Mild transforms only** (±10° rotation, 0.70–1.45× scale). Large
  rotations and extreme rescaling are out of distribution.
- **Category imbalance** follows COCO: person 8.1%, bird 7.5%, clock
  7.2% are the most frequent of 64 categories.
- **Source correlation**: 4,050 samples reuse a source object that
  another sample from the same image also uses. Group by `image_id`
  when constructing splits to avoid leakage.
- **Single copy-move per image**; no multi-source or nested forgeries.

## License and attribution

Released under **CC BY 4.0**.

Images derive from MS-COCO 2017. COCO *annotations* are CC BY 4.0;
COCO *images* originate from Flickr and remain subject to their owners'
terms — the COCO Consortium does not hold their copyright. The CC BY
4.0 grant covers the forgery generation, ground-truth masks and
metadata, not the underlying photographic content. Review the
[COCO terms of use](https://cocodataset.org/#termsofuse) before
redistributing the imagery, particularly for commercial use.

## Citation

```bibtex
@dataset{jain_coco_cmfd_2026,
  author    = {Jain, Harshita},
  title     = {{COCO-CMFD}: A Synthetic Copy-Move Forgery Dataset
               with Source/Target Ground Truth},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {TODO},
  url       = {https://github.com/harshitajain523/COCO-Copy-Move-Forgery-Dataset}
}
```

Please also cite MS-COCO:

```bibtex
@inproceedings{lin2014microsoft,
  title     = {Microsoft {COCO}: Common Objects in Context},
  author    = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and
               Hays, James and Perona, Pietro and Ramanan, Deva and
               Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle = {ECCV},
  year      = {2014}
}
```
