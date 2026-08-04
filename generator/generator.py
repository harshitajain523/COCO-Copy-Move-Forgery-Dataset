import cv2
import json
import logging
import numpy as np
import os
import random

from scene_layout import (
    build_surface_map,
    get_dominant_surface,
    heuristic_surface_code,
    is_support_compatible,
    is_surface_compatible,
    support_surface_code,
)
from patch_utils import pad_to_patch_grid, trimap_to_patch_labels

logger = logging.getLogger(__name__)

# ── Project-root-relative SAM mask directory ─────────────────────
# Resolved at import time so it is correct regardless of cwd.
# Must match precompute_sam_masks.SAM_MASKS_ROOT.
_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GEN_DIR)
SAM_MASKS_ROOT = os.path.join(_PROJECT_ROOT, "output", "sam_masks")

# ── Category constants ───────────────────────────────────────────
# Ground-type COCO category IDs: people, vehicles, animals, kitchen
# items, food, furniture — anything that should never appear floating
# in the sky. Flying categories (bird=16, airplane=5, kite=38) are
# explicitly excluded.
GROUND_CATS = frozenset({
    1,   # person
    2, 3, 4, 6, 7, 8,    # bicycle, car, motorcycle, bus, train, truck
    9, 10, 11,            # boat, traffic light, fire hydrant
    13, 14, 15,           # stop sign, parking meter, bench
    17, 18, 19, 20, 21, 22, 23, 24, 25,  # animals (cat→bear)
    27, 28,              # backpack, umbrella
    31, 32, 33,          # handbag, tie, suitcase
    39, 40, 41, 42, 43,  # bottle, wine glass, cup, fork, knife
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # spoon→sandwich
    56, 57, 58, 59, 60,  # broccoli→hot dog
    61, 62, 63, 64, 65,  # pizza→cake
    67, 70,              # chair, toilet
    72, 73, 74, 75, 76, 77, 78, 79, 80,  # tv→scissors
    81, 82, 84, 85, 86, 87, 88, 89, 90,  # hair drier→toothbrush
})

VEHICLE_CATS = frozenset({2, 3, 4, 6, 7, 8})       # bicycle→truck
LARGE_ANIMAL_CATS = frozenset({22, 23, 24, 25})    # elephant→giraffe

# Categories that carry text, digits, or strongly chiral structure.
# Horizontal flips produce mirrored writing (e.g. "POTS" stop signs,
# backwards clock faces) — an instant giveaway to a human observer.
NO_FLIP_CATS = frozenset({
    3, 5, 6, 7, 8,       # car, airplane, bus, train, truck (livery)
    10, 13, 14,          # traffic light, stop sign, parking meter
    72, 73, 76, 77,      # tv, laptop, keyboard, cell phone
    84, 85,              # book, clock
})

# Rectangular categories that legitimately fill their bounding box.
# For every other (organic) category, a mask that nearly fills the
# bbox means the segmentation failed and grabbed background.
BOXY_CATS = frozenset({
    13, 33,              # stop sign, suitcase
    72, 73, 76, 77,      # tv, laptop, keyboard, cell phone
    78, 79, 80, 82,      # microwave, oven, toaster, refrigerator
    84, 85,              # book, clock
})

# Rigid man-made categories that are always plumb in real photos:
# signs, mounted fixtures, furniture, appliances, vehicles. Even a
# small rotation (a tilted clock on a wall, a leaning bench) is a
# human-visible giveaway. Organic categories (people, animals, food)
# keep the mild rotation range.
NO_ROTATE_CATS = frozenset({
    2, 3, 4, 6, 7, 8, 9,    # bicycle→truck, boat
    10, 11, 13, 14, 15,     # traffic light, hydrant, stop sign,
                            # parking meter, bench
    62, 63, 65, 67, 70,     # chair, couch, bed, dining table, toilet
    72, 73, 76, 77,         # tv, laptop, keyboard, cell phone
    78, 79, 80, 81, 82,     # microwave, oven, toaster, sink, fridge
    84, 85,                 # book, clock
})

class CopyMoveGenerator:
    def __init__(
        self,
        output_dir_tampered,
        output_dir_masks,
        min_area=1000,
        max_area_ratio=0.15,
        min_area_ratio=0.02,
        max_target_area_ratio=0.20,
        min_laplacian_var=25.0,
        blend_mode="paste",
        feather_radius=0,
        semantic_radius_px=160,
        semantic_bg_color_thresh=55.0,
        hsv_hist_intersect_thresh=0.35,
        luminance_delta_thresh=45.0,
        padding_px=12,
        min_resolution=400,
        min_target_compactness=0.04,
        # Source-selection strictness. The placement and plausibility
        # gates are deliberately not parameterised: they stay constant
        # across every profile.
        edge_margin_px=15,
        isolation_dilation_px=2,
        min_fill_ratio=0.25,
        gate_profile="strict",
        transform_policy="coverage_like",
        flip_prob=0.5,
        k_ann=3,
        k_transform=3,
        k_dest=40,
        # Post-processing (global)
        jpeg_prob=0.6,
        jpeg_quality_min=65,
        jpeg_quality_max=95,
        noise_prob=0.3,
        noise_sigma_min=1.5,
        noise_sigma_max=8.0,
        bc_prob=0.25,
        contrast_min=0.85,
        contrast_max=1.15,
        brightness_max_abs=15,
        blur_prob=0.15,
        # Patch-level photometric
        patch_hsv_shift_prob=0.35,
        patch_v_min=0.88,
        patch_v_max=1.12,
        patch_s_min=0.90,
        patch_s_max=1.10,
        # Soft target mask output
        soft_target_prob=0.5,
        soft_sigma_min=1.0,
        soft_sigma_max=3.0,
        soft_alpha_min=0.93,
        soft_alpha_max=1.0,
    ):
        self.output_dir_tampered = output_dir_tampered
        self.output_dir_masks = output_dir_masks
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.max_target_area_ratio = max_target_area_ratio
        self.min_laplacian_var = min_laplacian_var
        self.blend_mode = blend_mode
        self.feather_radius = feather_radius
        self.semantic_radius_px = int(semantic_radius_px)
        self.semantic_bg_color_thresh = float(semantic_bg_color_thresh)
        self.hsv_hist_intersect_thresh = float(hsv_hist_intersect_thresh)
        self.luminance_delta_thresh = float(luminance_delta_thresh)
        self.padding_px = int(padding_px)
        self.min_resolution = int(min_resolution)
        self.min_target_compactness = float(min_target_compactness)
        self.edge_margin_px = int(edge_margin_px)
        self.isolation_dilation_px = int(isolation_dilation_px)
        self.min_fill_ratio = float(min_fill_ratio)
        self.gate_profile = str(gate_profile)
        self.transform_policy = str(transform_policy)
        self.flip_prob = float(flip_prob)
        self.k_ann = int(k_ann)
        self.k_transform = int(k_transform)
        self.k_dest = int(k_dest)

        self.jpeg_prob = float(jpeg_prob)
        self.jpeg_quality_min = int(jpeg_quality_min)
        self.jpeg_quality_max = int(jpeg_quality_max)
        self.noise_prob = float(noise_prob)
        self.noise_sigma_min = float(noise_sigma_min)
        self.noise_sigma_max = float(noise_sigma_max)
        self.bc_prob = float(bc_prob)
        self.contrast_min = float(contrast_min)
        self.contrast_max = float(contrast_max)
        self.brightness_max_abs = int(brightness_max_abs)
        self.blur_prob = float(blur_prob)

        self.patch_hsv_shift_prob = float(patch_hsv_shift_prob)
        self.patch_v_min = float(patch_v_min)
        self.patch_v_max = float(patch_v_max)
        self.patch_s_min = float(patch_s_min)
        self.patch_s_max = float(patch_s_max)

        self.soft_target_prob = float(soft_target_prob)
        self.soft_sigma_min = float(soft_sigma_min)
        self.soft_sigma_max = float(soft_sigma_max)
        self.soft_alpha_min = float(soft_alpha_min)
        self.soft_alpha_max = float(soft_alpha_max)

        # Ensure directories exist
        os.makedirs(output_dir_tampered, exist_ok=True)
        os.makedirs(output_dir_masks, exist_ok=True)
        os.makedirs(
            os.path.join(output_dir_masks, "soft_targets"),
            exist_ok=True,
        )
        # Enriched output directories
        self.output_dir_binary = os.path.join(
            os.path.dirname(output_dir_masks), "binary_masks_flat"
        )
        self.output_dir_patch_labels = os.path.join(
            os.path.dirname(output_dir_masks), "patch_labels"
        )
        self.output_dir_hard_neg = os.path.join(
            os.path.dirname(output_dir_masks), "hard_negatives"
        )
        for d in (
            self.output_dir_binary,
            self.output_dir_patch_labels,
            self.output_dir_hard_neg,
        ):
            os.makedirs(d, exist_ok=True)

    def extract_object_patch(self, img, ann, coco):
        """Extract the cropped RGB patch, cropped binary mask, bbox, and full binary mask.

        SAM mask paths resolve against SAM_MASKS_ROOT, an absolute
        project-root-relative constant, so the lookup is correct
        regardless of the calling process's working directory.
        """
        sam_path = os.path.join(
            SAM_MASKS_ROOT,
            f"{ann['image_id']}_{ann['id']}.png",
        )
        # annToMask returns a Fortran-ordered array; OpenCV requires
        # C-contiguous buffers (drawContours/grabCut raise otherwise).
        poly_mask = np.ascontiguousarray(coco.annToMask(ann))
        full_mask = poly_mask
        is_sam = False
        if os.path.exists(sam_path):
            sam_raw = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
            if (
                sam_raw is not None
                and sam_raw.shape[:2] == poly_mask.shape[:2]
            ):
                sam_bin = (sam_raw > 0).astype(np.uint8)
                # SAM sanity gate: bbox-prompted SAM sometimes grabs
                # background (walls, ledges) instead of the object.
                # Only trust it when it broadly agrees with the COCO
                # polygon; otherwise fall back to polygon + GrabCut.
                inter = int(np.logical_and(sam_bin, poly_mask).sum())
                union = int(np.logical_or(sam_bin, poly_mask).sum())
                if union > 0 and inter / union >= 0.60:
                    full_mask = sam_bin
                    is_sam = True

        x, y, w, h = map(int, ann['bbox'])
        
        # Safety checks
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return None, None, None
            
        # Crop mask and image (bbox slices are views — force
        # C-contiguous copies so every OpenCV call downstream works)
        try:
            cropped_mask = np.ascontiguousarray(
                full_mask[y:y+h, x:x+w]
            ).astype(np.uint8)
            cropped_img = np.ascontiguousarray(img[y:y+h, x:x+w])
        except Exception:
            return None, None, None
            
        if cropped_mask.size == 0 or cropped_img.size == 0:
            return None, None, None

        full_mask_bin = (full_mask > 0).astype(np.uint8)
        cropped_mask_bin = (cropped_mask > 0).astype(np.uint8)

        # Refine coarse COCO polygon boundaries using GrabCut
        # This removes the "blocky" background chunks from the object mask.
        if not is_sam and cropped_mask_bin.shape[0] > 10 and cropped_mask_bin.shape[1] > 10:
            gc_mask = np.where(cropped_mask_bin > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
            # Create a "sure foreground" core by eroding the mask
            kernel = np.ones((5,5), np.uint8)
            sure_fg = cv2.erode(cropped_mask_bin, kernel, iterations=2)
            gc_mask[sure_fg > 0] = cv2.GC_FGD
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(cropped_img, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
                refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

                # Accept the refinement only if it stays close to the
                # polygon (IoU >= 0.75). A pure area check let GrabCut
                # both eat objects down to fragments and inflate them
                # into amorphous background blobs.
                inter = int(np.logical_and(refined_mask, cropped_mask_bin).sum())
                union = int(np.logical_or(refined_mask, cropped_mask_bin).sum())
                if union > 0 and inter / union >= 0.75:
                    cropped_mask_bin = refined_mask
                    # Also update full_mask_bin to reflect this eroded boundary
                    full_mask_bin[y:y+h, x:x+w] = refined_mask
            except Exception:
                pass

        contours, hierarchy = cv2.findContours(cropped_mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            filled_holes = False
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1:
                    # Fill interior holes on the contiguous crop, then
                    # sync the full mask (drawing directly into the
                    # full_mask_bin slice view fails: non-contiguous).
                    cv2.drawContours(cropped_mask_bin, contours, i, 1, -1)
                    filled_holes = True
            if filled_holes:
                full_mask_bin[y:y+h, x:x+w] = cropped_mask_bin

        h_img, w_img = img.shape[:2]
        # Edge truncation.
        # Objects touching or very close to the image edge are often
        # truncated. Pasting a truncated object in the middle of the image
        # leaves an obvious, unnatural straight cut.
        margin = self.edge_margin_px
        if x <= margin or y <= margin or (x + w) >= (w_img - margin) or (y + h) >= (h_img - margin):
            return None, None, None

        # Bounding-box fill ratio.
        # If the mask occupies very little of its bounding box, the object
        # is likely heavily occluded, spindly, or disjointed.
        mask_area = int(cropped_mask_bin.sum())
        bbox_area = w * h
        fill_ratio = mask_area / float(max(bbox_area, 1))
        if fill_ratio < self.min_fill_ratio:
            return None, None, None
        # Inverse check: an organic object whose mask fills ~the whole
        # bbox means the mask leaked into the background (e.g. a SAM
        # bbox prompt returning the wall behind a dog).
        if (
            fill_ratio > 0.92
            and int(ann.get("category_id", -1)) not in BOXY_CATS
        ):
            return None, None, None

        # Source compactness.
        # Compute isoperimetric ratio of the source mask. Fragments
        # and heavily occluded objects have very low compactness (jagged).
        # We require a baseline of 0.15 for the source object.
        if self._compactness(cropped_mask_bin) < 0.15:
            return None, None, None

        return (cropped_img, cropped_mask_bin), (x, y, w, h), full_mask_bin

    @staticmethod
    def _expanded_warp(img_or_mask, M, out_w, out_h, is_mask=False):
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
        border_value = 0 if is_mask else (0, 0, 0)
        return cv2.warpAffine(
            img_or_mask,
            M,
            (out_w, out_h),
            flags=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    @staticmethod
    def _laplacian_var(bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _alpha_feather(mask_u8_0_255, radius):
        if radius <= 0:
            return mask_u8_0_255
        k = int(radius) * 2 + 1
        blurred = cv2.GaussianBlur(mask_u8_0_255, (k, k), 0)
        return blurred

    @staticmethod
    def _mean_color_bgr(img_bgr, mask_bin):
        """Mean BGR over mask_bin>0 pixels. Returns None if empty."""
        if mask_bin is None:
            return None
        ys, xs = np.where(mask_bin > 0)
        if len(ys) == 0:
            return None
        pix = img_bgr[ys, xs].astype(np.float32)
        return pix.mean(axis=0)

    @staticmethod
    def _ring_background_mask(mask_bin, ring_px=8):
        """Return a binary mask of background pixels in a ring around object."""
        k = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), np.uint8)
        obj = (mask_bin > 0).astype(np.uint8)
        dil = cv2.dilate(obj, k, iterations=1)
        ring = (dil > 0) & (obj == 0)
        return ring.astype(np.uint8)

    @staticmethod
    def _clamp(val, lo, hi):
        return max(lo, min(hi, val))

    @staticmethod
    def _hsv_hist(img_bgr, mask_u8_0_1):
        """Hue+sat histogram vector for mask pixels; returns None if empty."""
        if mask_u8_0_1 is None or not np.any(mask_u8_0_1 > 0):
            return None
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = (mask_u8_0_1.astype(np.uint8) * 255)
        h_hist = cv2.calcHist([hsv], [0], mask, [18], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], mask, [8], [0, 256])
        h_hist = cv2.normalize(h_hist, None).flatten()
        s_hist = cv2.normalize(s_hist, None).flatten()
        return np.concatenate([h_hist, s_hist]).astype(np.float32)

    @staticmethod
    def _hist_intersect(a, b):
        if a is None or b is None:
            return None
        return float(np.minimum(a, b).sum())

    @staticmethod
    def _compactness(mask_u8_0_1):
        """Isoperimetric ratio in [0,1]; lower => jaggier / degenerate."""
        probe = (mask_u8_0_1 > 0).astype(np.uint8)
        contours, _ = cv2.findContours(probe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        per = float(cv2.arcLength(c, True))
        if per <= 0:
            return 0.0
        return float(np.clip((4.0 * np.pi * area) / (per * per), 0.0, 1.0))

    def _sample_transform(self):
        policies = {
            "trans_only": {"scale": (1.0, 1.0), "rot": (0.0, 0.0)},
            "trans_scale": {"scale": (0.85, 1.15), "rot": (0.0, 0.0)},
            "coverage_like": {"scale": (0.85, 1.15), "rot": (-10.0, 10.0)},
            "full_aug": {"scale": (0.75, 1.25), "rot": (-45.0, 45.0)},
        }
        p = policies.get(self.transform_policy, policies["coverage_like"])
        scale_factor = random.uniform(p["scale"][0], p["scale"][1])
        rotation_angle = random.uniform(p["rot"][0], p["rot"][1])
        do_flip = random.random() < self.flip_prob
        return scale_factor, rotation_angle, do_flip

    @staticmethod
    def perspective_scale(src_y, dst_y, h_img, base_range=(0.85, 1.15)):
        """
        Adjust scale factor based on vertical position delta.

        Objects lower in frame = closer to camera = larger.
        Objects higher in frame = farther away = smaller.

        The sign convention is ``(dst_y - src_y)``: positive when the
        destination is below the source (nearer, so larger), negative
        when it is above (farther, so smaller).

        Parameters
        ----------
        src_y : float
            Source object center Y coordinate (pixels from top).
        dst_y : float
            Destination center Y coordinate (pixels from top).
        h_img : int
            Image height in pixels.
        base_range : tuple
            (min_scale, max_scale) for random perceptual jitter.

        Returns
        -------
        float
            Perspective-corrected scale factor.
        """
        # Positive delta → destination is LOWER (closer) → scale up.
        # Negative delta → destination is HIGHER (farther) → scale down.
        delta = (dst_y - src_y) / max(h_img, 1)
        perspective_factor = 1.0 + 0.5 * delta
        jitter = random.uniform(base_range[0], base_range[1])
        return float(np.clip(perspective_factor * jitter, 0.55, 1.45))

    @staticmethod
    def get_supercategory_pool(annotations, source_ann, coco):
        """
        Build candidate placement pool using COCO supercategories.

        Priority: same-category > same-supercategory > all annotations.

        Parameters
        ----------
        annotations : list
            All suitable annotations for this image.
        source_ann : dict
            The source annotation being copied.
        coco : pycocotools.coco.COCO
            COCO API instance.

        Returns
        -------
        tuple
            (candidate_annotations, placement_tier)
        """
        src_cat_id = source_ann.get("category_id", -1)
        src_id = source_ann.get("id", -1)

        # Get source supercategory
        try:
            src_cat_info = coco.loadCats(src_cat_id)[0]
            src_super = src_cat_info.get("supercategory", "unknown")
        except (IndexError, KeyError):
            src_super = "unknown"

        # Tier 1: same category
        tier1 = [
            a for a in annotations
            if a.get("category_id") == src_cat_id
            and a.get("id") != src_id
        ]
        if tier1:
            return tier1, "same_category"

        # Tier 2: same supercategory
        tier2 = []
        for a in annotations:
            if a.get("id") == src_id:
                continue
            try:
                cat_info = coco.loadCats(a["category_id"])[0]
                if cat_info.get("supercategory") == src_super:
                    tier2.append(a)
            except (IndexError, KeyError):
                continue
        if tier2:
            return tier2, "same_supercategory"

        # Tier 3: all other annotations
        tier3 = [
            a for a in annotations if a.get("id") != src_id
        ]
        if tier3:
            return tier3, "vertical_band_fallback"

        return [], "no_candidates"

    @staticmethod
    def build_hard_negative_mask(annotations, source_ann, coco, img_shape):
        """
        Build mask of all non-source COCO objects (hard negatives).

        These are genuine scene objects the contrastive loss must
        learn to distinguish from copies.

        Parameters
        ----------
        annotations : list
            All annotations for this image.
        source_ann : dict
            The source annotation being copied.
        coco : pycocotools.coco.COCO
            COCO API instance.
        img_shape : tuple
            (height, width) of the image.

        Returns
        -------
        np.ndarray
            (H, W) uint8 binary mask of hard negative regions.
        """
        h, w = img_shape[:2]
        neg_mask = np.zeros((h, w), dtype=np.uint8)
        src_id = source_ann.get("id", -1)
        for ann in annotations:
            if ann.get("id") == src_id:
                continue
            try:
                m = coco.annToMask(ann)
                mh, mw = m.shape[:2]
                crop_h = min(mh, h)
                crop_w = min(mw, w)
                neg_mask[:crop_h, :crop_w][
                    m[:crop_h, :crop_w] > 0
                ] = 1
            except Exception:
                continue
        return neg_mask

    @staticmethod
    def build_exclusion_mask(annotations, source_ann, coco,
                             img_shape):
        """
        Build a semantic exclusion mask.
        Excludes:
        1. Objects of the exact same category
        2. Objects of the same supercategory (e.g., prevents car on bus)
        3. Strong foreground objects (person, animal, vehicle)

        Parameters
        ----------
        annotations : list
            All annotations for this image.
        source_ann : dict
            The source annotation (excluded from the mask).
        coco : pycocotools.coco.COCO
            COCO API instance.
        img_shape : tuple
            (height, width) of the image.

        Returns
        -------
        np.ndarray
            (H, W) uint8 binary mask where 1 = occupied by same-category pixel.
        """
        h, w = img_shape[:2]
        exclusion = np.zeros((h, w), dtype=np.uint8)
        src_id = source_ann.get("id", -1)
        src_cat = source_ann.get("category_id", -1)
        
        try:
            src_supercat = coco.loadCats([src_cat])[0].get("supercategory", "")
        except Exception:
            src_supercat = ""

        strong_foreground = {"person", "vehicle", "animal"}

        for ann in annotations:
            if ann.get("id") == src_id:
                continue
                
            ann_cat = ann.get("category_id", -1)
            exclude_it = False
            
            if ann_cat == src_cat:
                exclude_it = True
            else:
                try:
                    ann_supercat = coco.loadCats([ann_cat])[0].get("supercategory", "")
                    if ann_supercat == src_supercat:
                        exclude_it = True
                    elif ann_supercat in strong_foreground:
                        exclude_it = True
                except Exception:
                    pass

            if not exclude_it:
                continue
                
            try:
                m = coco.annToMask(ann)
                mh, mw = m.shape[:2]
                crop_h = min(mh, h)
                crop_w = min(mw, w)
                exclusion[:crop_h, :crop_w][
                    m[:crop_h, :crop_w] > 0
                ] = 1
            except Exception:
                continue
        return exclusion

    @staticmethod
    def compute_occupied_fraction(occupied_mask, obj_mask_bin,
                                  y1, x1):
        """
        Compute what fraction of the paste region overlaps occupied
        pixels (other annotated objects).

        Parameters
        ----------
        occupied_mask : np.ndarray
            (H, W) binary mask of all non-source objects.
        obj_mask_bin : np.ndarray
            (pat_h, pat_w) binary mask of the object to paste.
        y1, x1 : int
            Top-left corner of placement in image coordinates.

        Returns
        -------
        float
            Fraction of paste pixels that land on occupied regions.
        """
        pat_h, pat_w = obj_mask_bin.shape[:2]
        y2 = y1 + pat_h
        x2 = x1 + pat_w
        h, w = occupied_mask.shape[:2]

        # Bounds check
        if y2 > h or x2 > w or y1 < 0 or x1 < 0:
            return 1.0

        occ_roi = occupied_mask[y1:y2, x1:x2]
        paste_pixels = obj_mask_bin > 0
        total_paste = int(paste_pixels.sum())
        if total_paste == 0:
            return 1.0

        overlap = int((occ_roi[paste_pixels] > 0).sum())
        return overlap / total_paste

    def _apply_patch_hsv_shift(self, obj_bgr):
        """Random V/S scaling of the copy patch (prob-gated).

        Stage 1 sets patch_hsv_shift_prob=0.0 → exact no-op.
        Returns (patch, applied_flag).
        """
        if random.random() >= self.patch_hsv_shift_prob:
            return obj_bgr, 0
        hsv = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = random.uniform(self.patch_s_min, self.patch_s_max)
        v = random.uniform(self.patch_v_min, self.patch_v_max)
        hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * v, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), 1

    def _post_process_composite(self, img_bgr):
        """Stage 2 degradations applied to the FINAL composite.

        Photometric only — nothing here may move pixels, so the
        ground-truth masks stay pixel-exact. Stage 1 sets every
        probability to 0.0 → exact no-op. Returns (image, params).
        """
        pp = {"jpeg_quality": None, "noise_sigma": None,
              "blur": 0, "contrast": None, "brightness": None}
        if random.random() < self.bc_prob:
            c = random.uniform(self.contrast_min, self.contrast_max)
            b = random.randint(
                -self.brightness_max_abs, self.brightness_max_abs
            )
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=c, beta=b)
            pp["contrast"], pp["brightness"] = round(c, 3), b
        if random.random() < self.blur_prob:
            img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
            pp["blur"] = 1
        if random.random() < self.noise_prob:
            sigma = random.uniform(
                self.noise_sigma_min, self.noise_sigma_max
            )
            # Seed numpy from the (already seeded) python RNG so the
            # noise field is reproducible per image.
            nrng = np.random.default_rng(random.getrandbits(63))
            noise = nrng.normal(0.0, sigma, img_bgr.shape)
            img_bgr = np.clip(
                img_bgr.astype(np.float32) + noise.astype(np.float32),
                0, 255,
            ).astype(np.uint8)
            pp["noise_sigma"] = round(sigma, 2)
        if random.random() < self.jpeg_prob:
            q = random.randint(
                self.jpeg_quality_min, self.jpeg_quality_max
            )
            ok, buf = cv2.imencode(
                ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q]
            )
            if ok:
                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                pp["jpeg_quality"] = q
        return img_bgr, pp

    def _make_soft_target(self, target_mask_u8_0_1):
        if random.random() >= self.soft_target_prob:
            return None
        sigma = random.uniform(self.soft_sigma_min, self.soft_sigma_max)
        alpha = random.uniform(self.soft_alpha_min, self.soft_alpha_max)
        soft = target_mask_u8_0_1.astype(np.float32)
        soft = cv2.GaussianBlur(soft, (0, 0), sigma)
        soft = np.clip(soft * alpha, 0.0, 1.0)
        return (soft * 255.0).astype(np.uint8)

    def generate(self, image_info, image_path, annotations, coco,
                 stuff_coco=None, surface_map=None,
                 use_perspective_scale=False,
                 use_supercategory_pool=False,
                 exclude_ann_ids=None, banned_cats=None,
                 out_suffix=""):
        """Attempt to generate a copy-move forged image.

        Parameters
        ----------
        image_info : dict
            COCO image metadata (must have 'id' and 'file_name').
        image_path : str
            Path to the source image file.
        annotations : list
            COCO annotations for this image.
        coco : pycocotools.coco.COCO
            COCO API instance for instance annotations.
        stuff_coco : pycocotools.coco.COCO, optional
            COCO API instance for stuff annotations.
        surface_map : np.ndarray, optional
            Pre-computed (H, W) surface classification map.
        use_perspective_scale : bool
            Use depth-aware scaling based on vertical position.
        use_supercategory_pool : bool
            Expand candidate pool via COCO supercategories.
        exclude_ann_ids : set, optional
            Annotation ids already used as sources for this image
            (multi-variant generation must pick a different object).
        banned_cats : set, optional
            Category ids that have hit their dataset quota.
        out_suffix : str
            Suffix appended to output basenames (e.g. '_v1') so
            multiple variants of one image don't collide.

        Returns
        -------
        dict or str
            Metadata dict on success, 'SKIP: reason' on failure.
        """
        img = cv2.imread(image_path)
        if img is None:
            return "SKIP: imread_failed"

        h_img, w_img = img.shape[:2]
        if min(h_img, w_img) < self.min_resolution:
            return "SKIP: low_resolution"
        img_area = h_img * w_img

        # Reject grayscale images: Stage 1 targets colour statistics.
        if len(img.shape) == 2 or (
            np.array_equal(img[:, :, 0], img[:, :, 1]) and 
            np.array_equal(img[:, :, 1], img[:, :, 2])
        ):
            return "SKIP: grayscale_image"

        # Build surface map if stuff annotations available but no map
        if stuff_coco is not None and surface_map is None:
            surface_map = build_surface_map(
                stuff_coco, image_info["id"], (h_img, w_img)
            )

        # Filter suitable annotations (not too small, not too big)
        suitable_anns = [
            ann for ann in annotations
            if self.min_area
            < float(ann.get("area", 0.0))
            < (img_area * self.max_area_ratio)
            and float(ann.get("area", 0.0)) > (img_area * self.min_area_ratio)
        ]
        if not suitable_anns:
            return "SKIP: no_suitable_annotations"

        # Multi-try: attempt multiple annotations/transforms/destinations
        last_skip = "SKIP: all_tries_exhausted"
        ann_pool = [
            a for a in suitable_anns
            if not (exclude_ann_ids and a.get("id") in exclude_ann_ids)
            and not (banned_cats
                     and a.get("category_id") in banned_cats)
        ]
        if not ann_pool:
            return "SKIP: no_suitable_annotations"
        random.shuffle(ann_pool)
        ann_pool = ann_pool[: max(1, min(self.k_ann, len(ann_pool)))]

        for ann in ann_pool:
            patch_data, src_bbox, src_mask_full = \
                self.extract_object_patch(img, ann, coco)
            if patch_data is None:
                last_skip = "SKIP: patch_extract_failed"
                continue

            cropped_img, cropped_mask_bin = patch_data
            src_x, src_y, src_w, src_h = src_bbox
            src_center_y = src_y + src_h / 2.0
            src_cat = int(ann.get("category_id", -1))

            # Filter out bad source supercategories (e.g., ties on faces)
            try:
                cat_info = coco.loadCats([src_cat])[0]
                if cat_info.get("supercategory") in {"accessory", "appliance", "electronic"}:
                    last_skip = f"SKIP: unwanted_supercategory ({cat_info.get('supercategory')})"
                    continue
            except Exception:
                pass

            # Source texture + minimum tamper size
            if not np.any(cropped_mask_bin > 0):
                last_skip = "SKIP: empty_object_pixels"
                continue
            if self._laplacian_var(cropped_img) < self.min_laplacian_var:
                last_skip = "SKIP: low_texture_source"
                continue
            src_area = int((src_mask_full > 0).sum())
            if src_area < int(self.min_area_ratio * img_area):
                last_skip = "SKIP: tamper_too_small"
                continue

            # Spatial isolation.
            # Objects that touch other annotations are usually occluded
            # (e.g., person riding a horse, baby on dad's shoulders,
            # or a person holding a tennis racket). We dilate the source
            # mask by 5 pixels and ensure it does NOT intersect with any
            # other annotation in the image. This guarantees we only copy
            # visually complete, standalone objects.
            hard_neg_mask = self.build_hard_negative_mask(
                annotations, ann, coco, (h_img, w_img)
            )
            r = self.isolation_dilation_px
            if r > 0:
                kernel_iso = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
                dilated_src = cv2.dilate(
                    src_mask_full, kernel_iso, iterations=1
                )
            else:
                dilated_src = src_mask_full
            if np.any((dilated_src > 0) & (hard_neg_mask > 0)):
                last_skip = "SKIP: object_not_isolated"
                continue

            # Source surface type (for surface compatibility check)
            src_surface = 0
            if surface_map is not None:
                src_surface = get_dominant_surface(
                    surface_map, src_x, src_y, src_w, src_h
                )
            elif stuff_coco is None:
                src_surface = heuristic_surface_code(
                    int(src_center_y), h_img
                )

            # Precompute donor background descriptors (ring)
            src_obj_crop = img[
                src_y:src_y + src_h, src_x:src_x + src_w
            ]
            src_ring = self._ring_background_mask(
                cropped_mask_bin, ring_px=8
            )
            src_bg_mean = self._mean_color_bgr(src_obj_crop, src_ring)
            src_bg_hist = self._hsv_hist(src_obj_crop, src_ring)

            # Build exclusion mask: ALL OTHER annotations of the SAME category
            # This prevents pasting a person ON TOP of another person
            exclusion_mask = self.build_exclusion_mask(
                annotations, ann, coco, (h_img, w_img)
            )

            # Source center for minimum separation enforcement
            src_center_x = src_x + src_w / 2.0

            # Build candidate placement pool (supercategory-aware)
            placement_tier = "random"
            candidate_centers = []
            if use_supercategory_pool:
                pool, placement_tier = self.get_supercategory_pool(
                    suitable_anns, ann, coco
                )
            else:
                pool = [
                    a for a in suitable_anns
                    if int(a.get("category_id", -2)) == src_cat
                    and a.get("id") != ann.get("id")
                ]
                if pool:
                    placement_tier = "same_category"

            random.shuffle(pool)
            for a in pool[:25]:
                x, y, w, h = map(int, a["bbox"])
                cx = x + w / 2.0
                cy = y + h / 2.0
                for _ in range(3):
                    jx = random.randint(
                        -self.semantic_radius_px,
                        self.semantic_radius_px,
                    )
                    jy = random.randint(
                        -self.semantic_radius_px,
                        self.semantic_radius_px,
                    )
                    candidate_centers.append(
                        (int(cx + jx), int(cy + jy))
                    )

            # Category-specific Y-floors (pixels from top must be ABOVE
            # this value for the destination center to be valid).
            # Rationale: vehicles and large animals sit firmly on the ground
            # and the sky in outdoor scenes often fills the top 50%+ of frame.
            # A single 40% threshold is too permissive for trains/cars.
            if src_cat in VEHICLE_CATS:
                # Vehicles are always firmly on the ground: 55% floor
                _ABS_Y_FLOOR = int(0.55 * h_img)
            elif src_cat in LARGE_ANIMAL_CATS:
                # Large animals also need a tighter constraint: 50%
                _ABS_Y_FLOOR = int(0.50 * h_img)
            else:
                # People, food, small objects, furniture: 42% floor
                # (slightly tighter than the previous 40%)
                _ABS_Y_FLOOR = int(0.42 * h_img)

            # Support surface the source object rests on (what is
            # directly under its footprint). This is the anchor for
            # the gravity gate applied to every candidate destination.
            src_support = 0
            if surface_map is not None:
                src_support = support_surface_code(
                    surface_map, src_mask_full[
                        src_y:src_y + src_h, src_x:src_x + src_w
                    ],
                    x0=src_x, y0=src_y,
                )

            # Relaxation ladder for vertical placement.
            # Tier 1 (strict): +-15% of src_center_y
            # Tier 2 (relaxed): +-30% of src_center_y
            # Tier 3 (permissive): no vertical constraint — only for
            # non-ground categories (birds, planes, kites). Ground
            # objects with no valid in-band destination are skipped
            # rather than pasted at an arbitrary height.
            _HORIZ_TIERS = [
                ("strict",     0.15),
                ("relaxed",    0.30),
            ]
            if src_cat not in GROUND_CATS:
                _HORIZ_TIERS.append(("permissive", None))

            for _ in range(max(1, self.k_transform)):
                scale_factor, rotation_angle, do_flip = self._sample_transform()

                # Never mirror text-bearing / chiral categories:
                # a flipped STOP sign or clock face is an instant
                # human-visible giveaway.
                if src_cat in NO_FLIP_CATS:
                    do_flip = False
                # Rigid mounted/man-made objects are always plumb.
                if src_cat in NO_ROTATE_CATS:
                    rotation_angle = 0.0

                # Compute the perspective scale, then gate on minimum
                # pixel area to reject invisible or floating objects.
                if use_perspective_scale:
                    if candidate_centers:
                        approx_dst_cy = random.choice(candidate_centers)[1]
                    else:
                        approx_dst_cy = random.randint(
                            h_img // 4, h_img - h_img // 4
                        )
                    scale_factor = self.perspective_scale(
                        src_center_y, approx_dst_cy, h_img
                    )

                    # Estimate post-scale area. If perspective reduction
                    # takes the object below min_area pixels, skip this
                    # transform: the result would be an invisible or
                    # anomalously tiny object.
                    estimated_area = (
                        float(ann.get("area", 1)) * scale_factor ** 2
                    )
                    if estimated_area < self.min_area:
                        last_skip = "SKIP: perspective_scale_too_small"
                        logger.debug(
                            "Skipping: perspective scale %.2f → "
                            "estimated_area %.0f < min_area %d",
                            scale_factor, estimated_area, self.min_area,
                        )
                        continue

                    # Absolute Y-floor: a ground-type object's approximate
                    # destination must lie at or below the horizon floor.
                    # Catches the case where perspective_scale correctly
                    # shrinks the object but the candidate height is still
                    # in sky territory.
                    if src_cat in GROUND_CATS and approx_dst_cy < _ABS_Y_FLOOR:
                        last_skip = "SKIP: ground_object_above_horizon"
                        logger.debug(
                            "Skipping: ground cat %d dst_cy=%d < floor=%d",
                            src_cat, approx_dst_cy, _ABS_Y_FLOOR,
                        )
                        continue

                # Padding around crop before warp.
                pad = max(0, self.padding_px)
                if pad > 0:
                    padded_img = cv2.copyMakeBorder(
                        cropped_img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101
                    )
                    padded_mask = cv2.copyMakeBorder(
                        cropped_mask_bin, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
                    )
                else:
                    padded_img, padded_mask = cropped_img, cropped_mask_bin

                if do_flip:
                    padded_img = cv2.flip(padded_img, 1)
                    padded_mask = cv2.flip(padded_mask, 1)

                base_h, base_w = padded_mask.shape[:2]
                # Expanded warp canvas so corners aren’t clipped.
                angle_rad = np.deg2rad(rotation_angle)
                cos_a = abs(np.cos(angle_rad)) * scale_factor
                sin_a = abs(np.sin(angle_rad)) * scale_factor
                pat_w = int(base_h * sin_a + base_w * cos_a)
                pat_h = int(base_h * cos_a + base_w * sin_a)
                if pat_w <= 0 or pat_h <= 0:
                    last_skip = "SKIP: invalid_patch_dims"
                    continue
                if pat_w >= w_img or pat_h >= h_img:
                    last_skip = "SKIP: patch_larger_than_image"
                    continue

                center = (base_w / 2.0, base_h / 2.0)
                M = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
                M[0, 2] += (pat_w - base_w) / 2.0
                M[1, 2] += (pat_h - base_h) / 2.0

                obj_rgb = self._expanded_warp(padded_img, M, pat_w, pat_h, is_mask=False)
                obj_mask_bin = self._expanded_warp(padded_mask, M, pat_w, pat_h, is_mask=True)
                if obj_mask_bin is None or not np.any(obj_mask_bin > 0):
                    last_skip = "SKIP: warped_mask_empty"
                    continue

                # Reject degenerate jaggy masks early.
                if self._compactness(obj_mask_bin) < self.min_target_compactness:
                    last_skip = "SKIP: low_boundary_quality"
                    continue

                target_area = int((obj_mask_bin > 0).sum())
                if target_area > int(self.max_target_area_ratio * img_area):
                    last_skip = "SKIP: target_too_large"
                    continue
                # Minimum size threshold.
                if target_area < int(self.min_area_ratio * img_area):
                    last_skip = "SKIP: tamper_too_small"
                    continue

                obj_rgb, patch_hsv_applied = \
                    self._apply_patch_hsv_shift(obj_rgb)
                obj_alpha = (obj_mask_bin * 255).astype(np.uint8)

                # Poisson-only dilation/border constraints.
                blend_mask = obj_alpha
                if self.blend_mode == "poisson":
                    kernel = np.ones((5, 5), np.uint8)
                    blend_mask = cv2.dilate(obj_alpha, kernel, iterations=1)
                    if (
                        blend_mask[0, :].any() or blend_mask[-1, :].any() or
                        blend_mask[:, 0].any() or blend_mask[:, -1].any()
                    ):
                        last_skip = "SKIP: mask_touches_patch_border"
                        continue

                placed = False

                # STRICT PLACEMENT: Boolean Cross-Correlation
                # Cross-correlate with the tightest horizon band first,
                # relaxing progressively through _HORIZ_TIERS only if no
                # valid spot is found.
                M_base = np.ones((h_img, w_img), dtype=np.float32)
                # Exclude ALL other COCO annotations
                M_base[exclusion_mask > 0] = 0.0
                # Exclude the source donor mask
                M_base[src_mask_full > 0] = 0.0

                # Semantic Lock (M_stuff)
                if surface_map is not None:
                    src_stuff_mask = get_dominant_surface(
                        surface_map, src_x, src_y, src_w, src_h
                    )
                    if src_stuff_mask > 0:
                        M_base[surface_map != src_stuff_mask] = 0.0

                # Try each relaxation tier in order
                dest_candidates = []
                accepted_horiz_tier = "none"
                kernel = obj_mask_bin.astype(np.float32)
                kernel_sum = float(np.sum(kernel))

                for _tier_name, _band_frac in _HORIZ_TIERS:
                    M_available = M_base.copy()
                    if _band_frac is not None:
                        # Apply horizon band mask
                        horizon_mask = np.zeros(
                            (h_img, w_img), dtype=np.float32
                        )
                        y_min = max(
                            0, int(src_center_y - _band_frac * h_img)
                        )
                        y_max = min(
                            h_img,
                            int(src_center_y + _band_frac * h_img),
                        )
                        horizon_mask[y_min:y_max, :] = 1.0
                        M_available *= horizon_mask

                    # Boolean cross-correlation
                    corr = cv2.matchTemplate(
                        M_available, kernel, cv2.TM_CCORR
                    )
                    valid_spots = np.where(corr >= kernel_sum - 1e-4)

                    if len(valid_spots[0]) > 0:
                        indices = list(zip(valid_spots[1], valid_spots[0]))
                        random.shuffle(indices)
                        for tx, ty in indices[:max(1, self.k_dest)]:
                            dst_cx = int(tx + pat_w / 2)
                            dst_cy = int(ty + pat_h / 2)
                            dest_candidates.append((dst_cx, dst_cy))
                        accepted_horiz_tier = _tier_name
                        logger.debug(
                            "Horizon tier '%s' yielded %d candidates",
                            _tier_name, len(dest_candidates),
                        )
                        break  # stop relaxing once we have candidates

                for (dst_cx, dst_cy) in dest_candidates:
                    dst_cx = self._clamp(
                        dst_cx, pat_w // 2, w_img - pat_w // 2
                    )
                    dst_cy = self._clamp(
                        dst_cy, pat_h // 2, h_img - pat_h // 2
                    )
                    dst_x = dst_cx - pat_w // 2
                    dst_y = dst_cy - pat_h // 2

                    # Canvas-clipping prevention.
                    # A destination box within 10px of any image edge produces
                    # an artificial straight line that the CNN uses as a cheat.
                    _EDGE_PAD = 10
                    if (
                        dst_x < _EDGE_PAD
                        or dst_y < _EDGE_PAD
                        or dst_x + pat_w > w_img - _EDGE_PAD
                        or dst_y + pat_h > h_img - _EDGE_PAD
                    ):
                        continue

                    # Final Y-floor gate on the exact destination.
                    # A spot that passed the horizon band can still end up
                    # above the floor once clamped, so ground-type objects
                    # are re-checked against the exact dst_cy.
                    if src_cat in GROUND_CATS and dst_cy < _ABS_Y_FLOOR:
                        continue

                    # PERSPECTIVE CONSISTENCY: scale_factor was derived
                    # from an *approximate* destination height, but the
                    # cross-correlation may have picked a spot far from
                    # it. Reject destinations where the applied scale no
                    # longer matches the depth implied by the actual
                    # dst_cy (allowing for the sampled jitter band).
                    if use_perspective_scale:
                        expected = 1.0 + 0.5 * (
                            (dst_cy - src_center_y) / max(h_img, 1)
                        )
                        ratio = scale_factor / max(expected, 1e-6)
                        if ratio < 0.80 or ratio > 1.25:
                            continue

                    # SUPPORT-SURFACE GATE (gravity check).
                    # The bbox-dominant surface check below is easily
                    # fooled: a toilet standing on a floor in front of a
                    # wall has a wall-dominated bbox, so wall pixels at
                    # mid-height pass as "compatible" and the object
                    # floats. Instead, compare what is directly under
                    # the pasted mask's footprint with what was under
                    # the source object's footprint.
                    if surface_map is not None and src_support > 0:
                        dst_support = support_surface_code(
                            surface_map, obj_mask_bin,
                            x0=dst_x, y0=dst_y,
                        )
                        if not is_support_compatible(
                            src_support, dst_support
                        ):
                            continue
                        # Ground-supported object over unlabeled pixels
                        # in the upper half of the frame → likely wall
                        # or sky gap in the stuff labels. Reject.
                        if (
                            dst_support == 0
                            and src_support == 1
                            and dst_cy < int(0.55 * h_img)
                        ):
                            continue

                    # Surface compatibility check
                    if surface_map is not None:
                        dst_surface = get_dominant_surface(
                            surface_map,
                            dst_x, dst_y, pat_w, pat_h,
                        )
                        # Primary surface compatibility check
                        if not is_surface_compatible(
                            src_surface, dst_surface
                        ):
                            continue
                        # Non-permissive fallback: even if the destination
                        # pixel is unlabeled (code 0 = other), a ground-type
                        # object must not be placed in the top 50% of the
                        # image when stuff_coco IS available.  This closes
                        # the 'other=compatible-with-anything' bypass that
                        # allows trains to land on unlabeled sky.
                        if (
                            dst_surface == 0
                            and src_cat in GROUND_CATS
                            and dst_cy < int(0.50 * h_img)
                        ):
                            continue
                    elif stuff_coco is None and src_surface > 0:
                        dst_heuristic = heuristic_surface_code(
                            dst_cy, h_img
                        )
                        if not is_surface_compatible(
                            src_surface, dst_heuristic
                        ):
                            continue

                    # Minimum separation from source center
                    separation = np.hypot(
                        dst_cx - src_center_x,
                        dst_cy - src_center_y,
                    )
                    min_sep = max(src_w, src_h) * 0.8
                    if separation < min_sep:
                        continue

                    # Pixel-level non-overlap with donor
                    y1, y2 = dst_y, dst_y + pat_h
                    x1, x2 = dst_x, dst_x + pat_w
                    src_slice = src_mask_full[y1:y2, x1:x2]
                    if (
                        src_slice.shape[0] != pat_h
                        or src_slice.shape[1] != pat_w
                    ):
                        continue
                    if np.any(
                        (src_slice > 0) & (obj_mask_bin > 0)
                    ):
                        continue

                    # Reject if pasting on top of another object of the SAME category
                    # Max 0% overlap allowed (protecting the pure forensic signal)
                    occ_frac = self.compute_occupied_fraction(
                        exclusion_mask, obj_mask_bin, dst_y, dst_x
                    )
                    if occ_frac > 0.0:
                        continue

                    # Reject a target overlapping the source region by >20%.
                    src_occ_frac = self.compute_occupied_fraction(
                        src_mask_full, obj_mask_bin, dst_y, dst_x
                    )
                    if src_occ_frac > 0.20:
                        last_skip = "SKIP: source_target_overlap_too_high"
                        continue

                    dest_roi = img[y1:y2, x1:x2]
                    if self._laplacian_var(dest_roi) < \
                            self.min_laplacian_var:
                        continue

                    # Background plausibility check
                    dest_ring = self._ring_background_mask(
                        obj_mask_bin, ring_px=8
                    )
                    dest_hist = self._hsv_hist(dest_roi, dest_ring)
                    sim = self._hist_intersect(
                        src_bg_hist, dest_hist
                    )
                    if (sim is not None
                            and sim < self.hsv_hist_intersect_thresh):
                        continue
                    dest_bg_mean = self._mean_color_bgr(
                        dest_roi, dest_ring
                    )
                    if sim is None and src_bg_mean is not None:
                        if dest_bg_mean is not None:
                            dist = float(np.linalg.norm(
                                dest_bg_mean - src_bg_mean
                            ))
                            if dist > self.semantic_bg_color_thresh:
                                continue

                    # LIGHTING CONSISTENCY: the HSV-histogram check
                    # above compares hue/saturation but is loose on
                    # brightness, so a sunlit object could land in a
                    # shadowed region (or vice versa). Compare mean
                    # luminance (BT.601) of the two background rings.
                    if (
                        src_bg_mean is not None
                        and dest_bg_mean is not None
                    ):
                        _lum = np.array([0.114, 0.587, 0.299])
                        lum_delta = abs(float(
                            (src_bg_mean - dest_bg_mean) @ _lum
                        ))
                        if lum_delta > self.luminance_delta_thresh:
                            continue

                    # BOUNDARY SMOOTHING: COCO polygon (and GrabCut)
                    # contours are coarse — visible straight segments
                    # and staircase jaggies. Gaussian-smooth the mask
                    # and re-binarise to round off the contour.
                    _smooth = cv2.GaussianBlur(
                        obj_mask_bin.astype(np.float32), (0, 0), 1.5
                    )
                    smooth_bin = (_smooth > 0.5).astype(np.uint8)

                    # HALO REMOVAL: erode the paste mask by 1px so the
                    # boundary fringe (a faint halo of the source's
                    # original background baked into segmentation
                    # edges) is left behind. The trimap below uses the
                    # same mask, so ground truth stays exact.
                    paste_mask = cv2.erode(
                        smooth_bin,
                        np.ones((3, 3), np.uint8),
                        iterations=1,
                    )
                    if not np.any(paste_mask > 0):
                        continue

                    # ── Compose ──────────────────────────────
                    mixed_clone = img.copy()
                    roi = mixed_clone[y1:y2, x1:x2]

                    if self.feather_radius > 0:
                        # MILD FEATHERED PASTE: alpha-composite over a
                        # ~feather_radius px band so the copy settles
                        # into the destination instead of a razor
                        # cutout. The interior stays a bit-exact copy;
                        # only the boundary band is mixed.
                        alpha = self._alpha_feather(
                            (paste_mask * 255).astype(np.uint8),
                            self.feather_radius,
                        ).astype(np.float32) / 255.0
                        # INWARD-ONLY FEATHER: zero the alpha outside
                        # paste_mask so blending never touches a pixel
                        # the trimap labels background. Every changed
                        # pixel is inside the GT mask (gate-1 IoU ~1.0).
                        alpha[paste_mask == 0] = 0.0
                        alpha3 = alpha[..., None]
                        comp = (
                            alpha3 * obj_rgb.astype(np.float32)
                            + (1.0 - alpha3) * roi.astype(np.float32)
                        )
                        roi = np.clip(comp, 0, 255).astype(np.uint8)
                    else:
                        # Hard paste: absolute pixel-to-pixel copy.
                        roi[paste_mask > 0] = obj_rgb[paste_mask > 0]

                    mixed_clone[y1:y2, x1:x2] = roi

                    # ── Stage 2 post-processing (no-op in Stage 1:
                    # all probabilities are 0). Photometric only, so
                    # the masks below stay pixel-exact.
                    mixed_clone, pp_meta = \
                        self._post_process_composite(mixed_clone)

                    # ── Tri-map ──────────────────────────────
                    tri_map = np.zeros(
                        (h_img, w_img), dtype=np.uint8
                    )
                    tri_map[src_mask_full > 0] = 128
                    tri_map[y1:y2, x1:x2][
                        paste_mask > 0
                    ] = 255

                    # ── Binary forgery mask ──────────────────
                    binary_mask = (tri_map > 0).astype(np.uint8)
                    binary_mask *= 255

                    # ── Patch-level labels (p=16) ────────────
                    tri_padded = pad_to_patch_grid(tri_map, 16)
                    patch_labels = trimap_to_patch_labels(
                        tri_padded, 16
                    )

                    # ── Hard negative mask ───────────────────
                    hard_neg = self.build_hard_negative_mask(
                        annotations, ann, coco, (h_img, w_img)
                    )

                    # ── Soft target (optional) ───────────────
                    soft = self._make_soft_target(
                        (tri_map == 255).astype(np.uint8)
                    )
                    soft_name = None
                    if soft is not None:
                        soft_name = (
                            f"{image_info['file_name']}"
                            .replace('.jpg', '')
                            + out_suffix
                            + "_soft_target.png"
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.output_dir_masks,
                                "soft_targets", soft_name,
                            ),
                            soft,
                        )

                    # ── Category metadata ────────────────────
                    cat_name = "unknown"
                    supercat = "unknown"
                    try:
                        cat_info = coco.loadCats(src_cat)[0]
                        cat_name = cat_info.get("name", "unknown")
                        supercat = cat_info.get(
                            "supercategory", "unknown"
                        )
                    except (IndexError, KeyError):
                        pass

                    # Count same-category instances
                    same_cat_count = sum(
                        1 for a in annotations
                        if a.get("category_id") == src_cat
                    )

                    # ── Save all outputs ─────────────────────
                    out_base = image_info['file_name'].replace(
                        '.jpg', ''
                    ) + out_suffix
                    img_fname = f"{out_base}.png"
                    mask_fname = f"{out_base}_mask.png"
                    binary_fname = f"{out_base}_binary.png"
                    patch_fname = f"{out_base}_patches.npy"
                    neg_fname = f"{out_base}_hard_neg.npy"

                    cv2.imwrite(
                        os.path.join(
                            self.output_dir_tampered, img_fname
                        ),
                        mixed_clone,
                    )
                    cv2.imwrite(
                        os.path.join(
                            self.output_dir_masks, mask_fname
                        ),
                        tri_map,
                    )
                    cv2.imwrite(
                        os.path.join(
                            self.output_dir_binary, binary_fname
                        ),
                        binary_mask,
                    )
                    np.save(
                        os.path.join(
                            self.output_dir_patch_labels,
                            patch_fname,
                        ),
                        patch_labels,
                    )
                    np.save(
                        os.path.join(
                            self.output_dir_hard_neg, neg_fname
                        ),
                        hard_neg,
                    )

                    # ── Enriched metadata ────────────────────
                    src_tex = float(
                        self._laplacian_var(cropped_img)
                    )
                    dst_tex = float(
                        self._laplacian_var(dest_roi)
                    )
                    output_data = {
                        'image_id': int(image_info['id']),
                        'image_filename': img_fname,
                        'mask_filename': mask_fname,
                        'binary_mask_filename': binary_fname,
                        'patch_labels_filename': patch_fname,
                        'hard_neg_filename': neg_fname,
                        'soft_target_filename': soft_name,
                        'transform_policy': self.transform_policy,
                        'flip': int(do_flip),
                        'blend_mode': self.blend_mode,
                        'scale_factor': round(scale_factor, 3),
                        'rotation_angle': round(
                            rotation_angle, 3
                        ),
                        'source_bbox': str(
                            (src_x, src_y, src_w, src_h)
                        ),
                        'target_bbox': str(
                            (dst_x, dst_y, pat_w, pat_h)
                        ),
                        'source_ann_id': ann.get('id', -1),
                        'variant': out_suffix or '_v0',
                        'source_category_id': src_cat,
                        'source_category_name': cat_name,
                        'source_supercategory': supercat,
                        'placement_tier': placement_tier,
                        # Horizon tier used (strict/relaxed/permissive),
                        # for downstream quality filtering.
                        'horizon_tier': accepted_horiz_tier,
                        'same_cat_instance_count': same_cat_count,
                        'source_texture_var': round(src_tex, 2),
                        'dest_texture_var': round(dst_tex, 2),
                        'bg_similarity_score': (
                            round(sim, 4)
                            if sim is not None else None
                        ),
                        'image_height': h_img,
                        'image_width': w_img,
                        # Which source-selection strictness produced
                        # this row: 'strict' or 'relaxed_v1' (recovery)
                        'gate_profile': self.gate_profile,
                        # Stage 2 post-processing provenance
                        # (all None/0 in Stage 1)
                        'patch_hsv_applied': patch_hsv_applied,
                        'jpeg_quality': pp_meta['jpeg_quality'],
                        'noise_sigma': pp_meta['noise_sigma'],
                        'blur': pp_meta['blur'],
                        'contrast': pp_meta['contrast'],
                        'brightness': pp_meta['brightness'],
                    }

                    logger.info(
                        "Generated %s | tier=%s cat=%s",
                        img_fname, placement_tier, cat_name,
                    )
                    return output_data

                last_skip = "SKIP: no_valid_destination"

        return last_skip

