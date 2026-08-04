"""
Stage configuration presets for COCO-CMFD dataset generation.

Stage 1 (clean):  Pure copy signal for contrastive pretraining.
                  Zero post-processing. Paste-only blending.
Stage 2 (synthetic): DeFaCTo-like synthetic complexity.
                     Full post-processing. Mixed blending.

Usage:
    from stage_config import get_stage_config
    cfg = get_stage_config("stage1")
    generator = CopyMoveGenerator(**cfg.to_generator_kwargs(
        output_dir_tampered="...",
        output_dir_masks="...",
    ))
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class StageConfig:
    """Immutable configuration preset for a generation stage."""

    # ── Identity ─────────────────────────────────────────────
    name: str
    description: str

    # ── Object selection ─────────────────────────────────────
    min_area: int = 1000
    max_area_ratio: float = 0.15
    min_area_ratio: float = 0.02
    max_target_area_ratio: float = 0.20
    min_laplacian_var: float = 25.0
    # Minimum image side. 320 admits VGA-class COCO images, which a
    # higher floor would discard in quantity.
    min_resolution: int = 320
    # Compactness floor. Below ~0.07, GrabCut-trimmed fragments
    # (very jagged, low isoperimetric ratio) pass through.
    min_target_compactness: float = 0.07
    # Source-selection strictness. Defaults = the strict profile the
    # main Stage 1 run used; the recovery preset relaxes them and
    # tags its rows via gate_profile so they remain separable.
    edge_margin_px: int = 15
    isolation_dilation_px: int = 2
    min_fill_ratio: float = 0.25
    gate_profile: str = "strict"

    # ── Blending ─────────────────────────────────────────────
    blend_mode: str = "paste"
    feather_radius: int = 0

    # ── Semantic placement ───────────────────────────────────
    semantic_radius_px: int = 160
    semantic_bg_color_thresh: float = 55.0
    hsv_hist_intersect_thresh: float = 0.35
    luminance_delta_thresh: float = 45.0
    padding_px: int = 12

    # ── Transforms ───────────────────────────────────────────
    transform_policy: str = "coverage_like"
    flip_prob: float = 0.5

    # ── Retry budget ─────────────────────────────────────────
    k_ann: int = 3
    k_transform: int = 3
    k_dest: int = 40

    # ── Post-processing (global) ─────────────────────────────
    jpeg_prob: float = 0.0
    jpeg_quality_min: int = 65
    jpeg_quality_max: int = 95
    noise_prob: float = 0.0
    noise_sigma_min: float = 1.5
    noise_sigma_max: float = 8.0
    bc_prob: float = 0.0
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    brightness_max_abs: int = 15
    blur_prob: float = 0.0

    # ── Patch-level photometric ──────────────────────────────
    patch_hsv_shift_prob: float = 0.0
    patch_v_min: float = 0.88
    patch_v_max: float = 1.12
    patch_s_min: float = 0.90
    patch_s_max: float = 1.10

    # ── Soft target mask ─────────────────────────────────────
    soft_target_prob: float = 0.0
    soft_sigma_min: float = 1.0
    soft_sigma_max: float = 3.0
    soft_alpha_min: float = 0.93
    soft_alpha_max: float = 1.0

    # ── Pipeline tunables ────────────────────────────────────
    num_images: int = 10
    subset_size: int = 200
    use_perspective_scale: bool = False
    use_stuff_annotations: bool = False
    use_supercategory_pool: bool = False
    # Max forgeries per source image (each uses a different source
    # object). >1 is required to reach 20k from COCO's 118k images
    # at strict-quality yield rates.
    variants_per_image: int = 1
    # Per-category share cap (fraction of num_images). Prevents
    # easy-to-isolate categories (clocks, toilets) from dominating.
    category_share_cap: float = 0.05

    def to_generator_kwargs(self, output_dir_tampered, output_dir_masks):
        """Convert to kwargs dict for CopyMoveGenerator.__init__."""
        # Exclude pipeline-level fields that aren't generator params
        exclude = {
            "name", "description", "num_images",
            "subset_size", "use_perspective_scale",
            "use_stuff_annotations", "use_supercategory_pool",
            "variants_per_image", "category_share_cap",
        }
        kwargs = {
            k: v for k, v in asdict(self).items()
            if k not in exclude
        }
        kwargs["output_dir_tampered"] = output_dir_tampered
        kwargs["output_dir_masks"] = output_dir_masks
        return kwargs


# ── Stage Presets ────────────────────────────────────────────────

STAGE1_CLEAN = StageConfig(
    name="stage1_clean",
    description=(
        "Pure copy signal for contrastive pretraining. "
        "Zero post-processing. The model learns what a copy IS."
    ),
    # Blending: direct paste with a mild 2px boundary feather —
    # the copy signal stays bit-exact in the interior, but the
    # razor-sharp cutout edge (a giveaway no real forger leaves)
    # is softened into the destination.
    blend_mode="paste",
    feather_radius=2,
    # Transforms: mild affine (the copy should be recognisable)
    transform_policy="coverage_like",
    flip_prob=0.3,
    # Post-processing: NONE — this is the whole point
    jpeg_prob=0.0,
    noise_prob=0.0,
    bc_prob=0.0,
    blur_prob=0.0,
    patch_hsv_shift_prob=0.0,
    soft_target_prob=0.0,
    # Semantic placement enhancements
    use_perspective_scale=True,
    use_stuff_annotations=True,
    use_supercategory_pool=True,
    # Retry budget: higher for clean data (more selective).
    # k_ann=12 ≈ "try every suitable annotation" for most images.
    k_ann=12,
    k_transform=4,
    k_dest=50,
    # Pipeline
    num_images=1000,
    subset_size=40000,
    # Each variant copies a different source object. An image that
    # passes the gates once will often support several distinct
    # forgeries.
    variants_per_image=4,
    category_share_cap=0.05,
)


STAGE2_SYNTHETIC = StageConfig(
    name="stage2_synthetic",
    description=(
        "DeFaCTo-like synthetic complexity. Full post-processing. "
        "Teaches the model that copies can be damaged."
    ),
    # Blending: paste only (Poisson hides too much for training)
    blend_mode="paste",
    feather_radius=3,
    # Transforms: aggressive
    transform_policy="full_aug",
    flip_prob=0.5,
    # Post-processing: full suite
    jpeg_prob=0.7,
    jpeg_quality_min=60,
    jpeg_quality_max=95,
    noise_prob=0.4,
    noise_sigma_min=1.5,
    noise_sigma_max=10.0,
    bc_prob=0.35,
    contrast_min=0.80,
    contrast_max=1.20,
    brightness_max_abs=20,
    blur_prob=0.25,
    # Patch photometric: stronger shifts
    patch_hsv_shift_prob=0.5,
    patch_v_min=0.80,
    patch_v_max=1.20,
    patch_s_min=0.85,
    patch_s_max=1.15,
    soft_target_prob=0.0,
    # Semantic placement enhancements
    use_perspective_scale=True,
    use_stuff_annotations=True,
    use_supercategory_pool=True,
    # Retry budget
    k_ann=5,
    k_transform=5,
    k_dest=50,
    # Pipeline
    num_images=10,
    subset_size=500,
)


import dataclasses as _dc

# Recovery pass: identical to Stage 1 except mildly relaxed
# SOURCE-selection gates (edge margin, isolation, fill floor).
# Placement/plausibility gates (support surface, horizon, perspective,
# no-flip, luminance) are untouched. Rows are tagged relaxed_v1.
STAGE1_RECOVERY = _dc.replace(
    STAGE1_CLEAN,
    name="stage1_recovery",
    description=(
        "Recovery pass over strict-run failures: relaxed source "
        "selection (edge 15→8px, isolation 5x5→3x3, fill 0.25→0.20), "
        "identical placement gates. Rows tagged gate_profile=relaxed_v1."
    ),
    edge_margin_px=8,
    isolation_dilation_px=1,
    min_fill_ratio=0.20,
    gate_profile="relaxed_v1",
)

# relaxed_v2 additionally lowers the minimum paste size from 2% to
# 1.2% of image area (the absolute 1000 px floor is unchanged). Small
# copy-moves are the realistic hard case in the CMFD literature, and
# most COCO objects fall below a 2% floor.
STAGE1_RECOVERY_V2 = _dc.replace(
    STAGE1_RECOVERY,
    name="stage1_recovery_v2",
    description=(
        "relaxed_v1 source gates + min paste size 2%→1.2% of image "
        "area. Placement gates identical to strict."
    ),
    min_area_ratio=0.012,
    gate_profile="relaxed_v2",
)


# ── Registry ────────────────────────────────────────────────────

_STAGES = {
    "stage1": STAGE1_CLEAN,
    "stage1_clean": STAGE1_CLEAN,
    "stage1_recovery": STAGE1_RECOVERY,
    "stage1_recovery_v2": STAGE1_RECOVERY_V2,
    "stage2": STAGE2_SYNTHETIC,
    "stage2_synthetic": STAGE2_SYNTHETIC,
}


def get_stage_config(stage_name, **overrides):
    """
    Retrieve a stage configuration by name with optional overrides.

    Parameters
    ----------
    stage_name : str
        One of 'stage1', 'stage1_clean', 'stage2', 'stage2_synthetic'.
    **overrides
        Any field to override in the preset.

    Returns
    -------
    StageConfig
        Frozen dataclass with the resolved configuration.
    """
    if stage_name not in _STAGES:
        valid = ", ".join(sorted(_STAGES.keys()))
        raise ValueError(
            f"Unknown stage '{stage_name}'. Valid: {valid}"
        )
    base = _STAGES[stage_name]
    if not overrides:
        return base
    # Create new instance with overrides applied
    fields = asdict(base)
    fields.update(overrides)
    return StageConfig(**fields)
