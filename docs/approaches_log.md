# COCO-CMFD — Approaches Log

> Running record of all design decisions, experiments, and results.
> Updated continuously as the pipeline evolves.

---

## Entry 1 — Original Pipeline (Pre-Refactor)

**Date**: 2026-04-15 → 2026-05-04
**Status**: ✅ Working, but misaligned with MemoryForger architecture

### What It Did
- Single-process, WSL-safe copy-move forgery generator
- Used MS COCO 2017 instance annotations as source objects
- Subprocess isolation for annotation loading (449 MB JSON)
- Applied post-processing: JPEG (60%), noise (30%), blur (15%),
  contrast/brightness (25%), patch HSV shifts (35%)
- Semantic placement via same-category proximity + vertical band
  constraint + HSV histogram background matching
- Output: tampered PNG + trimap mask (0/128/255) + metadata CSV

### Problems Identified
1. **Post-processing contradicts Stage 1 requirement** — architecture
   spec demands zero post-processing for pretraining data
2. **Semantic placement too shallow** — falls back to random when no
   same-category instances exist (majority of COCO images)
3. **No stuff-annotation awareness** — objects can be pasted on
   incompatible surfaces (car in sky)
4. **No perspective-aware scaling** — uniform random scale ignores depth
5. **Missing outputs for contrastive training** — no patch-level labels,
   no hard negative masks, no binary masks
6. **Scale**: only 5 images, needs 20,000

### Decision
Fork pipeline into Stage 1 (clean) and Stage 2 (augmented) via config
presets. Refactor semantic placement. Add enriched outputs.

---

## Entry 2 — Stage-Aware Refactor (Current Work)

**Date**: 2026-05-04
**Status**: 🔄 In progress

### Design Decisions

#### Stage Configuration System
- Created `stage_config.py` with frozen dataclass presets
- Stage 1: zero post-processing, paste-only blending, mild transforms
- Stage 2: full post-processing, mixed blending, aggressive transforms
- Each stage writes to its own output subdirectory

#### Semantic Placement Improvements
- **Supercategory expansion**: if no same-category instances, try
  same-supercategory, then compatible-vertical-band fallback
- **Stuff annotations**: use COCO stuff_train2017.json for surface-type
  awareness (ground/sky/water/wall classification)
- **Perspective scaling**: vertical displacement → rough depth proxy
  adjusts scale factor (lower = closer = larger)
- **Surface-constrained jitter**: candidate destinations checked against
  stuff surface map before acceptance
- **Multi-instance image preference**: score images by same-category
  instance count for built-in hard negatives

#### Output Enrichment
- Binary forgery mask (0/255) alongside trimap
- Patch-level labels (p=16 grid) saved as .npy
- Hard negative mask (all non-source COCO objects)
- Enriched metadata: category, supercategory, placement tier,
  quality metrics, hard negative bboxes

### Sanity Check Plan
- Generate 10 images Stage 1, visualize grid
- Generate 10 images Stage 2, visualize grid
- Manual inspection before scaling to 20k

---

## Entry 3 — [Next entry will be added after sanity check results]
