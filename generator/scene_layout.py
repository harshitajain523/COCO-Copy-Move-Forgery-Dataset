"""
Scene layout module for surface-aware semantic placement.

Uses COCO Stuff annotations to determine what surface type exists
at any pixel location (ground, sky, water, wall, etc.) and checks
whether a source → destination placement is semantically plausible.

Falls back to vertical-position heuristics when stuff annotations
are unavailable.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Surface classification groups ────────────────────────────────
# COCO Stuff category names → surface groups.
# Reference: https://github.com/nightrome/cocostuff/blob/master/labels.md

GROUND_SURFACES = frozenset({
    "floor-marble", "floor-other", "floor-stone", "floor-tile",
    "floor-wood", "gravel", "mud", "pavement", "platform",
    "playingfield", "railroad", "road", "sand", "snow",
    "dirt", "grass", "ground-other",
})

SKY_SURFACES = frozenset({
    "sky-other", "clouds",
})

WATER_SURFACES = frozenset({
    "water-other", "river", "sea", "waterdrops",
})

WALL_SURFACES = frozenset({
    "wall-brick", "wall-concrete", "wall-other", "wall-panel",
    "wall-stone", "wall-tile", "wall-wood", "fence",
    "building-other",
})

CEILING_SURFACES = frozenset({
    "ceiling-other", "ceiling-tile", "roof",
})

# Groups that are compatible with each other for placement
_SURFACE_GROUPS = [
    GROUND_SURFACES,
    SKY_SURFACES,
    WATER_SURFACES,
    WALL_SURFACES,
    CEILING_SURFACES,
]


def classify_surface(stuff_cat_name):
    """
    Classify a COCO Stuff category name into a surface group.

    Parameters
    ----------
    stuff_cat_name : str
        COCO Stuff category name (e.g., 'grass', 'sky-other').

    Returns
    -------
    str
        One of 'ground', 'sky', 'water', 'wall', 'ceiling', 'other'.
    """
    name = stuff_cat_name.lower().strip()
    if name in GROUND_SURFACES:
        return "ground"
    if name in SKY_SURFACES:
        return "sky"
    if name in WATER_SURFACES:
        return "water"
    if name in WALL_SURFACES:
        return "wall"
    if name in CEILING_SURFACES:
        return "ceiling"
    return "other"


def build_surface_map(stuff_coco, img_id, img_shape):
    """
    Build a per-pixel surface classification map for an image.

    Parameters
    ----------
    stuff_coco : pycocotools.coco.COCO
        COCO object loaded with stuff annotations.
    img_id : int
        COCO image ID.
    img_shape : tuple
        (height, width) of the image.

    Returns
    -------
    np.ndarray
        (H, W) uint8 array where each pixel has a surface code:
        0=other, 1=ground, 2=sky, 3=water, 4=wall, 5=ceiling.
    """
    h, w = img_shape[:2]
    surface_map = np.zeros((h, w), dtype=np.uint8)

    # Surface code mapping
    code_map = {
        "ground": 1, "sky": 2, "water": 3,
        "wall": 4, "ceiling": 5, "other": 0,
    }

    try:
        ann_ids = stuff_coco.getAnnIds(imgIds=img_id)
        anns = stuff_coco.loadAnns(ann_ids)
    except Exception as exc:
        logger.debug("No stuff annotations for image %d: %s", img_id, exc)
        return surface_map

    # Build category ID → surface code lookup
    cat_ids = stuff_coco.getCatIds()
    cats = stuff_coco.loadCats(cat_ids)
    cat_to_surface = {}
    for cat in cats:
        surface_type = classify_surface(cat["name"])
        cat_to_surface[cat["id"]] = code_map.get(surface_type, 0)

    # Paint each annotation onto the surface map
    for ann in anns:
        cat_id = ann.get("category_id", -1)
        surface_code = cat_to_surface.get(cat_id, 0)
        if surface_code == 0:
            continue  # Skip 'other' — only paint known surfaces
        try:
            mask = stuff_coco.annToMask(ann)
            # Handle size mismatches from annotation vs actual image
            mh, mw = mask.shape[:2]
            crop_h = min(mh, h)
            crop_w = min(mw, w)
            surface_map[:crop_h, :crop_w][
                mask[:crop_h, :crop_w] > 0
            ] = surface_code
        except Exception:
            continue

    return surface_map


def get_surface_at(surface_map, x, y):
    """
    Get the surface code at pixel (x, y).

    Returns
    -------
    int
        Surface code: 0=other, 1=ground, 2=sky, 3=water, 4=wall, 5=ceiling.
    """
    h, w = surface_map.shape[:2]
    if 0 <= y < h and 0 <= x < w:
        return int(surface_map[y, x])
    return 0


def get_dominant_surface(surface_map, x, y, w, h):
    """
    Get the dominant surface type in a bounding box region.

    Parameters
    ----------
    surface_map : np.ndarray
        (H, W) surface classification map.
    x, y, w, h : int
        Bounding box coordinates.

    Returns
    -------
    int
        Most frequent non-zero surface code in the region,
        or 0 if the region is entirely 'other'.
    """
    map_h, map_w = surface_map.shape[:2]
    y1 = max(0, y)
    y2 = min(map_h, y + h)
    x1 = max(0, x)
    x2 = min(map_w, x + w)

    if y2 <= y1 or x2 <= x1:
        return 0

    region = surface_map[y1:y2, x1:x2]
    # Count non-zero surface codes
    codes, counts = np.unique(region[region > 0], return_counts=True)
    if len(codes) == 0:
        return 0
    return int(codes[np.argmax(counts)])


def support_surface_code(surface_map, mask_bin, x0=0, y0=0,
                         probe_px=12):
    """
    Determine the surface an object is resting ON.

    For each column of the mask, find the lowest foreground pixel
    and sample the surface map in a thin band directly below it.
    The dominant non-zero code across all columns is the support
    surface. This is far more discriminative than the dominant
    surface of the bounding box: an object standing on a floor in
    front of a wall has bbox-dominant 'wall' but support 'ground'.

    Parameters
    ----------
    surface_map : np.ndarray
        (H, W) surface classification map.
    mask_bin : np.ndarray
        Binary object mask (any resolution patch).
    x0, y0 : int
        Top-left offset of the mask patch in image coordinates.
    probe_px : int
        Depth of the probe band below the object's bottom edge.

    Returns
    -------
    int
        Dominant surface code under the footprint, 0 if unknown.
    """
    map_h, map_w = surface_map.shape[:2]
    fg = mask_bin > 0
    if not fg.any():
        return 0

    codes = []
    cols = np.where(fg.any(axis=0))[0]
    # Sample up to ~32 columns evenly for speed
    if len(cols) > 32:
        cols = cols[np.linspace(0, len(cols) - 1, 32).astype(int)]
    for c in cols:
        rows = np.where(fg[:, c])[0]
        bottom = y0 + rows[-1]
        xx = x0 + c
        if not (0 <= xx < map_w):
            continue
        band = surface_map[
            min(bottom + 1, map_h - 1):
            min(bottom + 1 + probe_px, map_h),
            xx,
        ]
        band = band[band > 0]
        if band.size:
            codes.extend(band.tolist())

    if not codes:
        return 0
    vals, counts = np.unique(np.asarray(codes), return_counts=True)
    return int(vals[np.argmax(counts)])


def is_support_compatible(src_support, dst_support):
    """
    Check whether a destination support surface can plausibly hold
    an object whose source support was *src_support*.

    Unlike :func:`is_surface_compatible`, this is strict about
    gravity: a ground-supported object may only land on ground, a
    water-supported object only on water. Wall support (mounted
    objects like clocks and signs) transfers to walls only.
    Unknown (0) on either side defers to the caller's fallbacks.

    Returns
    -------
    bool
    """
    if src_support == 0 or dst_support == 0:
        return True  # unknown — caller applies Y-floor fallbacks
    if src_support == dst_support:
        return True
    # ground ↔ water is the only cross-pair we allow (shorelines,
    # boats, wading animals are common in COCO).
    if {src_support, dst_support} == {1, 3}:
        return True
    return False


def is_surface_compatible(src_surface_code, dst_surface_code):
    """
    Check if source and destination surfaces are compatible.

    Rules:
    - Same surface type → always compatible
    - ground ↔ ground → yes
    - sky ↔ sky → yes
    - ground ↔ sky → NO (the big win)
    - water ↔ ground → NO
    - other ↔ anything → yes (permissive fallback)

    Parameters
    ----------
    src_surface_code, dst_surface_code : int
        Surface codes (0=other, 1=ground, 2=sky, 3=water, 4=wall, 5=ceiling).

    Returns
    -------
    bool
    """
    # 'Other' is always compatible (permissive fallback)
    if src_surface_code == 0 or dst_surface_code == 0:
        return True
    # Same surface → always compatible
    if src_surface_code == dst_surface_code:
        return True
    # Ground ↔ wall is acceptable (objects can be near walls)
    if {src_surface_code, dst_surface_code} == {1, 4}:
        return True
    # Everything else: incompatible
    return False


# ── Heuristic fallback (no stuff annotations) ───────────────────

def heuristic_surface_code(y, h_img, is_outdoor_hint=True):
    """
    Estimate surface type from vertical position alone.

    Very rough heuristic: top 25% = sky, bottom 60% = ground,
    middle = other. Only applied when stuff annotations are absent.

    Parameters
    ----------
    y : int
        Vertical pixel position.
    h_img : int
        Image height.
    is_outdoor_hint : bool
        If False, skip sky heuristic (indoor scenes).

    Returns
    -------
    int
        Estimated surface code.
    """
    frac = y / max(h_img, 1)
    # Sky boundary widened from 25% to 35% to shrink the unlabelled
    # "other" dead zone where ground objects could slip through.
    if is_outdoor_hint and frac < 0.35:
        return 2  # sky
    # Ground threshold widened from 40% to 45% for the same reason.
    if frac > 0.45:
        return 1  # ground
    return 0  # other


# ── Surface code ↔ name mapping ─────────────────────────────────

SURFACE_NAMES = {
    0: "other",
    1: "ground",
    2: "sky",
    3: "water",
    4: "wall",
    5: "ceiling",
}


def surface_code_to_name(code):
    """Convert numeric surface code to human-readable name."""
    return SURFACE_NAMES.get(code, "unknown")
