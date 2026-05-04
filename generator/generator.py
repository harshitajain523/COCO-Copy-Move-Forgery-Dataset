import cv2
import itertools
import json
import logging
import numpy as np
import os
import random

from scene_layout import (
    build_surface_map,
    get_dominant_surface,
    heuristic_surface_code,
    is_surface_compatible,
    surface_code_to_name,
)
from patch_utils import pad_to_patch_grid, trimap_to_patch_labels

logger = logging.getLogger(__name__)

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
        padding_px=12,
        min_resolution=400,
        min_target_compactness=0.04,
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
        self.padding_px = int(padding_px)
        self.min_resolution = int(min_resolution)
        self.min_target_compactness = float(min_target_compactness)
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
        """Extract the cropped RGB patch, cropped binary mask, bbox, and full binary mask."""
        full_mask = coco.annToMask(ann)
        x, y, w, h = map(int, ann['bbox'])
        
        # Safety checks
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return None, None, None
            
        # Crop mask and image
        try:
            cropped_mask = full_mask[y:y+h, x:x+w].astype(np.uint8)
            cropped_img = img[y:y+h, x:x+w]
        except Exception:
            return None, None, None
            
        if cropped_mask.size == 0 or cropped_img.size == 0:
            return None, None, None

        full_mask_bin = (full_mask > 0).astype(np.uint8)
        cropped_mask_bin = (cropped_mask > 0).astype(np.uint8)

        # Refine coarse COCO polygon boundaries using GrabCut
        # This removes the "blocky" background chunks from the object mask.
        if cropped_mask_bin.shape[0] > 10 and cropped_mask_bin.shape[1] > 10:
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
                
                # Only use refined mask if it didn't completely destroy the object
                if (refined_mask > 0).sum() > (cropped_mask_bin > 0).sum() * 0.2:
                    cropped_mask_bin = refined_mask
                    # Also update full_mask_bin to reflect this eroded boundary
                    full_mask_bin[y:y+h, x:x+w] = refined_mask
            except Exception:
                pass

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
        Objects higher in frame = farther = smaller.

        Parameters
        ----------
        src_y : float
            Source object center Y coordinate.
        dst_y : float
            Destination center Y coordinate.
        h_img : int
            Image height.
        base_range : tuple
            (min_scale, max_scale) for random jitter.

        Returns
        -------
        float
            Perspective-corrected scale factor.
        """
        delta = (src_y - dst_y) / max(h_img, 1)
        perspective_factor = 1.0 + 0.5 * delta
        jitter = random.uniform(base_range[0], base_range[1])
        return float(np.clip(perspective_factor * jitter, 0.7, 1.4))

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
        if random.random() >= self.patch_hsv_shift_prob:
            return obj_bgr
        hsv = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] *= random.uniform(self.patch_v_min, self.patch_v_max)
        hsv[:, :, 1] *= random.uniform(self.patch_s_min, self.patch_s_max)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _post_process_composite(self, img_bgr):
        out = img_bgr
        if random.random() < self.jpeg_prob:
            q = random.randint(self.jpeg_quality_min, self.jpeg_quality_max)
            _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if random.random() < self.noise_prob:
            sigma = random.uniform(self.noise_sigma_min, self.noise_sigma_max)
            noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < self.bc_prob:
            alpha = random.uniform(self.contrast_min, self.contrast_max)
            beta = random.randint(-abs(self.brightness_max_abs), abs(self.brightness_max_abs))
            out = np.clip(alpha * out.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        if random.random() < self.blur_prob:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        return out

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
                 use_supercategory_pool=False):
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

        # Filter 1: Strictly reject grayscale images for Stage 1 color learning
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
        ann_pool = suitable_anns[:]
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

            # Category-aware vertical band
            ground_like = {
                1, 2, 3, 4, 6, 7, 8, 16, 17, 18, 19, 20
            }
            def _vertical_ok(cy):
                if src_cat in ground_like:
                    return abs(cy - src_center_y) <= 0.20 * h_img
                return True

            for _ in range(max(1, self.k_transform)):
                scale_factor, rotation_angle, do_flip = self._sample_transform()

                # Activating Depth Realism: Calculate perspective scale dynamically
                if use_perspective_scale:
                    if candidate_centers:
                        approx_dst_cy = random.choice(candidate_centers)[1]
                    else:
                        approx_dst_cy = random.randint(h_img // 4, h_img - h_img // 4)
                    scale_factor = self.perspective_scale(src_center_y, approx_dst_cy, h_img)

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
                # Filter 3: Minimum size threshold check
                if target_area < int(self.min_area_ratio * img_area):
                    last_skip = "SKIP: tamper_too_small"
                    continue

                obj_rgb = self._apply_patch_hsv_shift(obj_rgb)
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

                # Destination search (semantic-first then random)
                def _iter_centers():
                    for c in candidate_centers:
                        yield c
                    while True:
                        yield (
                            random.randint(
                                pat_w // 2, w_img - pat_w // 2
                            ),
                            random.randint(
                                pat_h // 2, h_img - pat_h // 2
                            ),
                        )

                placed = False
                dest_candidates = list(
                    itertools.islice(_iter_centers(), max(1, self.k_dest))
                )

                for (dst_cx, dst_cy) in dest_candidates:
                    if not _vertical_ok(dst_cy):
                        continue
                    dst_cx = self._clamp(
                        dst_cx, pat_w // 2, w_img - pat_w // 2
                    )
                    dst_cy = self._clamp(
                        dst_cy, pat_h // 2, h_img - pat_h // 2
                    )
                    dst_x = dst_cx - pat_w // 2
                    dst_y = dst_cy - pat_h // 2

                    # Surface compatibility check
                    if surface_map is not None:
                        dst_surface = get_dominant_surface(
                            surface_map,
                            dst_x, dst_y, pat_w, pat_h,
                        )
                        if not is_surface_compatible(
                            src_surface, dst_surface
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

                    # Filter 2: Reject if target overlaps the original SOURCE object > 20%
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
                    if sim is None and src_bg_mean is not None:
                        dest_bg_mean = self._mean_color_bgr(
                            dest_roi, dest_ring
                        )
                        if dest_bg_mean is not None:
                            dist = float(np.linalg.norm(
                                dest_bg_mean - src_bg_mean
                            ))
                            if dist > self.semantic_bg_color_thresh:
                                continue

                    # ── Compose ──────────────────────────────
                    mixed_clone = img.copy()
                    if self.blend_mode == "poisson":
                        try:
                            mixed_clone = cv2.seamlessClone(
                                obj_rgb, img, blend_mask,
                                (dst_cx, dst_cy),
                                cv2.NORMAL_CLONE,
                            )
                        except Exception:
                            last_skip = "SKIP: poisson_failed"
                            continue
                    else:
                        alpha = obj_alpha
                        if self.feather_radius > 0:
                            alpha = self._alpha_feather(
                                alpha, self.feather_radius
                            )
                        alpha_f = (
                            alpha.astype(np.float32) / 255.0
                        )[..., None]
                        roi = mixed_clone[
                            y1:y2, x1:x2
                        ].astype(np.float32)
                        src_f = obj_rgb.astype(np.float32)
                        mixed = (
                            src_f * alpha_f
                            + roi * (1.0 - alpha_f)
                        )
                        mixed_clone[y1:y2, x1:x2] = \
                            mixed.astype(np.uint8)

                    # Global post-processing
                    mixed_clone = self._post_process_composite(
                        mixed_clone
                    )

                    # ── Tri-map ──────────────────────────────
                    tri_map = np.zeros(
                        (h_img, w_img), dtype=np.uint8
                    )
                    tri_map[src_mask_full > 0] = 128
                    tri_map[y1:y2, x1:x2][
                        obj_mask_bin > 0
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
                    )
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
                        'source_category_id': src_cat,
                        'source_category_name': cat_name,
                        'source_supercategory': supercat,
                        'placement_tier': placement_tier,
                        'same_cat_instance_count': same_cat_count,
                        'source_texture_var': round(src_tex, 2),
                        'dest_texture_var': round(dst_tex, 2),
                        'bg_similarity_score': (
                            round(sim, 4)
                            if sim is not None else None
                        ),
                        'image_height': h_img,
                        'image_width': w_img,
                    }

                    logger.info(
                        "Generated %s | tier=%s cat=%s",
                        img_fname, placement_tier, cat_name,
                    )
                    return output_data

                last_skip = "SKIP: no_valid_destination"

        return last_skip

