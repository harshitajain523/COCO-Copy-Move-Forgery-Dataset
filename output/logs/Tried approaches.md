## Tried approaches (systematic log)

This document is a running “lab notebook” of what we tried while building the COCO→CMFD synthetic generator, what worked, what failed, and what we learned. The intent is to make the evolution of the pipeline auditable and to guide the next iterations toward a dataset that generalizes to **Coverage** / **CASIA** style copy-move forgeries.

---

### Context and constraints

- **Goal**: procedurally generate a large-scale copy-move forgery dataset from MS COCO (target scale eventually: ~50k images) with correct supervision masks and controlled difficulty for pretraining DL models.
- **Environment**: WSL2 with **8 GB RAM** allocated (from a 16 GB system). Frequent WSL crashes initially.
- **Important modeling preference**:
  - **Tri-map supervision** is preferred for pretraining: background=0, donor(source)=128, target=255.
  - The dataset is meant for **pretraining/generalization**, not necessarily photorealistic forgery simulation at first.
  - Priority: teach “copy & paste” and correspondence cues; realism can be introduced later.

---

## Attempt 0 — Baseline pipeline review (pre-change)

### What existed

- `generator/main.py`:
  - Loaded COCO annotations in main.
  - Built a giant list of tasks for **all ~118k COCO train2017 images**:
    - each task included `img_info`, `img_path`, and the full `anns` list.
  - Spawned `multiprocessing.Pool(processes=cpu_count())`.
  - Each worker also loaded COCO JSON via `COCO(annot_path)`.
- `generator/generator.py`:
  - Extracted object patch using `coco.annToMask(ann)` (full-size mask) then bbox crop.
  - Applied random scale/rotation and Poisson blending (`cv2.seamlessClone`).
  - Created tri-map, but:
    - called `annToMask` again (duplicate heavy operation),
    - allocated multiple full-image arrays (`target_mask_canvas`, `np.where(...).astype(...)`).

### What went wrong

- **WSL crashes** were consistent with **memory exhaustion**:
  - The **task list** construction pinned huge amounts of Python objects in RAM.
  - Workers spawned equal to CPU count: high process count → high memory footprint.
  - COCO JSON and index structures were loaded in **every worker** (duplicated N times).
- **Quality issues** in outputs:
  - Tri-maps were often not faithful to what was blended (mask dilation vs mask GT mismatch; alpha-based target GT not perfectly consistent).
  - Destination non-overlap constraint was bbox-based (not mask-based).
  - Rotations clipped patches because `warpAffine` output size stayed `(src_w, src_h)`.
  - Some tampered regions were too tiny to be meaningful for training.

### What we learned

- For WSL stability:
  - Avoid eager global task lists.
  - Cap worker count.
  - Avoid huge pickled payloads.
  - Reduce per-task peak allocations.
- For supervision quality:
  - GT must be derived from the same binary mask actually used to paste/blend, not a loosely related alpha.

---

## Attempt 1 — Stabilize WSL + reduce memory spikes (main orchestration)

### Change summary

Edits in `generator/main.py`:

- **Stream tasks lazily** (iterator/generator) instead of building a full list for all images.
- **Cap workers** with new `--workers` option (default: `min(4, cpu_count())`).
- **Recycle workers** using `--maxtasksperchild` to reduce long-lived fragmentation from numpy/OpenCV.
- **Reduce IPC payload**:
  - Instead of passing full `anns` per image from main → workers,
  - pass `(img_id, img_path, file_name)` and load annotations inside the worker from `_worker_coco`.
- **Limit OpenCV threads per process**: `cv2.setNumThreads(1)` in worker init.

### Result

- WSL crash frequency dropped significantly.
- Dataset generation ran reliably for small targets (10 images).
- However: `maxtasksperchild` too low caused frequent worker respawns (re-loading COCO in workers repeatedly) → slowdowns. We learned to use larger values (e.g., 50) during normal runs.

### What we learned

- `maxtasksperchild` is a trade-off:
  - too low → repeated expensive COCO init,
  - too high → potential long-lived memory fragmentation.
- WSL stability depends more on **controlling parallelism** and **avoiding huge in-memory lists** than anything else.

---

## Attempt 2 — Fix GT fidelity + geometry correctness (generator core)

### Change summary

Edits in `generator/generator.py` to make tri-map match the actual transformation/paste:

- **Compute source mask once** and reuse it.
- **Expand warp canvas**:
  - compute `pat_w/pat_h` based on rotation + scale extents,
  - warp RGB patch into expanded canvas,
  - warp **binary mask** with `INTER_NEAREST` into expanded canvas.
- Use the **warped binary mask**:
  - to build `obj_alpha` for paste,
  - and to define **target=255 GT**.
- **Pixel-level non-overlap**:
  - ensure pasted target mask does not overlap donor mask in the original image.
- Reduce unnecessary full-size allocations in tri-map creation.

### Result

- Tri-map became **pixel-faithful**:
  - source=128 came from COCO segmentation mask,
  - target=255 came from warped binary mask used for paste/blend,
  - no mismatch due to dilation vs GT.
- Placement correctness improved because destination computations used `pat_w/pat_h`.

### What we learned

- Using alpha derived from interpolated RGB warps can be noisy; warping the binary mask using nearest-neighbor is a more faithful GT strategy.
- Correctness and training utility improve when GT precisely matches the pixels that were pasted.

---

## Attempt 3 — Shift from “realistic blending” to “pretraining copy-paste”

### Motivation

For pretraining, realism is not the first priority. The model must learn the core copy-move signal:

- duplicated content,
- structured donor/target supervision,
- boundaries that can be obvious (like Defacto-synthetic examples).

### Change summary

Added controls to `generator/generator.py` + CLI in `generator/main.py`:

- **Blend mode switch**:
  - `--blend-mode paste` (default) uses hard copy-paste,
  - `--blend-mode poisson` retains `seamlessClone` path for later realism phases.
- Optional **edge feathering**:
  - `--feather-radius` (default 0) allows mild edge smoothing when desired.
- **Tamper size control**:
  - `--min-area-ratio` (default 0.02) prevents tiny edits,
  - `--max-target-area-ratio` (default 0.20) prevents huge edits.
- **Low-texture avoidance**:
  - `--min-laplacian-var` rejects low-texture source and destination regions (proxy for sky/water/flat surfaces).

### Result

- Outputs became more “copy-paste obvious”.
- Tamper regions became more meaningful as a fraction of the image (not dominated by tiny blobs).

### What we learned

- “Too much blending” can teach the wrong inductive bias for pretraining. Hard paste is often preferable initially.
- Texture constraints reduce nonsense placements but increase failure rate (trade-off).

---

## Attempt 4 — Make destinations semantically plausible

### Motivation

To generalize to Coverage/CASIA, the destination should “make sense”:

- person should appear near other people at similar ground level,
- objects should paste onto similar background material (cheap proxy for semantics).

### Change summary

Added **semantic destination heuristics**:

- Prefer candidate destinations near **other instances of the same COCO category** (within `--semantic-radius-px`).
- For `person` (COCO category id = 1), enforce destination vertical proximity to reduce “floating”.
- Add **background ring mean-color similarity** between donor and destination (thresholded by `--semantic-bg-color-thresh`).

### Result

- More plausible placements in many cases.
- Still imperfect, but noticeably reduced “object pasted into totally different material”.

### What we learned

- Simple heuristics (same-category proximity + background ring similarity) go surprisingly far without deep models.
- Heuristics increase the number of rejections; to scale, the generator must attempt multiple candidates per image (next step).

---

## Attempt 5 — Category index and scale-minded sampling (“top objects”)

### Motivation

At 50k scale, we need generation to:

- avoid rare/weird categories that produce poor training signal,
- be tunable and reproducible,
- have transparent failure diagnostics.

### Change summary

In `generator/main.py`:

- Build a **category frequency index** from COCO `dataset["annotations"]`.
- Allow generation to use only:
  - top-N categories: `--top-categories N` (default 30), or
  - explicit allowlist: `--category-ids "1,3,8,..."`.
- Filter per-image annotations in workers to allowed categories.

Diagnostics improvements:

- Introduced `SKIP: <reason>` returns from `CopyMoveGenerator.generate()`.
- Main process aggregates **failure reason counts** and writes:
  - summary counts, and
  - a small sample of failures
  - to `output/logs/failed_samples.txt`.

### Result (30-image run; “quality-first” settings)

Command:

`python generator/main.py --num-images 30 --workers 2 --maxtasksperchild 50 --blend-mode paste --feather-radius 0 --min-area-ratio 0.02 --min-laplacian-var 25 --semantic-radius-px 200 --semantic-bg-color-thresh 50 --top-categories 25`

Observed:

- Generated **30** successes with **258** failures (high rejection ratio, expected for strict filters).
- Dominant failure reasons (counts during that run):
  - `SKIP: no_suitable_annotations` (many images had no allowed-category annotations passing area gates),
  - `SKIP: tamper_too_small` (source object too small relative to image),
  - `SKIP: mask_touches_patch_border` (warped+dilated mask touches patch border),
  - `SKIP: no_valid_destination` (semantic + texture + overlap constraints too strict for that image).
- Mask sanity across the 30 outputs:
  - tri-map values were strictly `{0,128,255}`
  - source area ratio min/median/max ≈ `0.020 / 0.048 / 0.149`
  - target area ratio min/median/max ≈ `0.019 / 0.046 / 0.127`

### What we learned

- We now have an interpretable “knob surface”:
  - failure reasons tell exactly what to fix for yield without harming quality.
- The largest yield killers are:
  - “no suitable annotation” and “tamper too small”
  - followed by “mask touches patch border” and “no valid destination”

---

## Current status (what works)

- **WSL stability**: significantly improved via streaming tasks, limiting workers, smaller IPC, and OpenCV thread control.
- **GT correctness**: tri-map is now derived from:
  - COCO segmentation for donor,
  - warped binary mask used for paste for target.
- **Pretraining focus**: default operation is hard copy-paste (not Poisson blending).
- **Semantic plausibility**: improved via same-category proximity + background ring similarity.
- **Scale readiness**: failure reasons are measurable; category filtering prevents “junk” categories from dominating.

---

## Current status (what’s still failing / missing)

### 1) Yield is low under strict “quality-first” constraints

This is acceptable during development but must be improved for 50k scale. The next algorithmic step is to **increase attempts per image** rather than relaxing constraints:

- try multiple annotations per image (not a single random pick),
- try multiple transform proposals per annotation,
- try multiple semantic + random destinations per transform,
- stop early when success found.

### 2) Transform policy is not yet “Coverage-like” by default

If we want Coverage/CASIA generalization, we likely want a curriculum or a default that heavily prefers:

- translation,
- mild scale,
- small rotations (optional),
instead of the current wide `[-45,+45]` degrees.

### 3) Better “semantic plausibility” beyond mean-color rings

Mean-color ring similarity is cheap but imperfect. Potential future improvements:

- gradient histogram similarity for donor/destination rings,
- HSV histogram similarity (illumination robustness),
- exclude sky/water-like regions using simple color/texture priors,
- category-specific placement policies (person→ground band; vehicles→road-like surfaces).

### 4) Mask border issue (`mask_touches_patch_border`)

This is currently a major skip reason, caused by warping + dilation producing a mask that touches the patch edge. Ways forward:

- add padding margin around the warped patch canvas,
- reduce dilation for paste-mode (since we’re not Poisson blending),
- adaptively re-sample rotation/scale if mask touches border.

---

## Next planned improvements (recommended)

1) **Multi-try strategy** (biggest yield win without lowering quality)
   - For each image, sample K annotations and K transforms each, instead of one.
2) **Pretraining transform modes**
   - Add `--transform-policy` with presets: `trans_only`, `trans_scale`, `trans_scale_rot_small`.
3) **Category policy refinement**
   - Default top-N is good; add optional “priority” set (e.g., person/vehicle/sign) with weights.
4) **Paste-mode boundary policy**
   - Since we’re in paste-mode, disable dilation requirement entirely, and use only the binary mask.

---

## Reproducibility notes

- Use `--seed` to fix the randomized image traversal.
- For stable performance in WSL (8 GB), recommended starting settings:
  - `--workers 2`
  - `--maxtasksperchild 50`

