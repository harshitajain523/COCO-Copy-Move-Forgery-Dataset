import cv2
import numpy as np
import random
import os

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
        os.makedirs(os.path.join(output_dir_masks, "soft_targets"), exist_ok=True)

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

    def generate(self, image_info, image_path, annotations, coco):
        """Attempts to generate a copy-move forged image.

        Returns:
          - dict on success
          - 'SKIP: <reason>' on controlled failure
          - 'ERROR: ...' is handled by caller for unexpected exceptions
        """
        img = cv2.imread(image_path)
        if img is None:
            return "SKIP: imread_failed"
            
        h_img, w_img = img.shape[:2]
        if min(h_img, w_img) < self.min_resolution:
            return "SKIP: low_resolution"
        img_area = h_img * w_img

        # Filter suitable annotations (not too small, not too big)
        suitable_anns = [
            ann for ann in annotations
            if self.min_area < float(ann.get("area", 0.0)) < (img_area * self.max_area_ratio)
        ]
        if not suitable_anns:
            return "SKIP: no_suitable_annotations"

        # Multi-try: attempt multiple annotations/transforms/destinations.
        last_skip = "SKIP: all_tries_exhausted"
        ann_pool = suitable_anns[:]
        random.shuffle(ann_pool)
        ann_pool = ann_pool[: max(1, min(self.k_ann, len(ann_pool)))]

        for ann in ann_pool:
            patch_data, src_bbox, src_mask_full = self.extract_object_patch(img, ann, coco)
            if patch_data is None:
                last_skip = "SKIP: patch_extract_failed"
                continue

            cropped_img, cropped_mask_bin = patch_data
            src_x, src_y, src_w, src_h = src_bbox
            src_center_y = src_y + src_h / 2.0
            src_cat = int(ann.get("category_id", -1))

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

            # Precompute donor background descriptors (ring).
            src_obj_crop = img[src_y:src_y + src_h, src_x:src_x + src_w]
            src_ring = self._ring_background_mask(cropped_mask_bin, ring_px=8)
            src_bg_mean = self._mean_color_bgr(src_obj_crop, src_ring)
            src_bg_hist = self._hsv_hist(src_obj_crop, src_ring)

            # Candidate centers near same-category instances.
            candidate_centers = []
            if src_cat != -1:
                same_cat = [
                    a for a in suitable_anns
                    if int(a.get("category_id", -2)) == src_cat and a.get("id") != ann.get("id")
                ]
                random.shuffle(same_cat)
                for a in same_cat[:25]:
                    x, y, w, h = map(int, a["bbox"])
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    for _ in range(3):
                        jx = random.randint(-self.semantic_radius_px, self.semantic_radius_px)
                        jy = random.randint(-self.semantic_radius_px, self.semantic_radius_px)
                        candidate_centers.append((int(cx + jx), int(cy + jy)))

            # Category-aware vertical band (extend person rule to ground-like classes).
            ground_like = {1, 2, 3, 4, 6, 7, 8, 16, 17, 18, 19, 20}  # person/vehicles/animals (subset)
            def _vertical_ok(cy):
                if src_cat in ground_like:
                    return abs(cy - src_center_y) <= 0.20 * h_img
                return True

            for _ in range(max(1, self.k_transform)):
                scale_factor, rotation_angle, do_flip = self._sample_transform()

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

                # Destination search (semantic-first then random).
                def _iter_centers():
                    for c in candidate_centers:
                        yield c
                    while True:
                        yield (
                            random.randint(pat_w // 2, w_img - pat_w // 2),
                            random.randint(pat_h // 2, h_img - pat_h // 2),
                        )

                import itertools
                placed = False
                for (dst_center_x, dst_center_y) in itertools.islice(_iter_centers(), max(1, self.k_dest)):
                    if not _vertical_ok(dst_center_y):
                        continue
                    dst_center_x = self._clamp(dst_center_x, pat_w // 2, w_img - pat_w // 2)
                    dst_center_y = self._clamp(dst_center_y, pat_h // 2, h_img - pat_h // 2)
                    dst_x = dst_center_x - pat_w // 2
                    dst_y = dst_center_y - pat_h // 2

                    # Pixel-level non-overlap with donor.
                    y1, y2 = dst_y, dst_y + pat_h
                    x1, x2 = dst_x, dst_x + pat_w
                    src_slice = src_mask_full[y1:y2, x1:x2]
                    if src_slice.shape[0] != pat_h or src_slice.shape[1] != pat_w:
                        continue
                    if np.any((src_slice > 0) & (obj_mask_bin > 0)):
                        continue

                    dest_roi = img[y1:y2, x1:x2]
                    if self._laplacian_var(dest_roi) < self.min_laplacian_var:
                        continue

                    # Background plausibility: HSV hist + fallback mean-color.
                    dest_ring = self._ring_background_mask(obj_mask_bin, ring_px=8)
                    dest_hist = self._hsv_hist(dest_roi, dest_ring)
                    sim = self._hist_intersect(src_bg_hist, dest_hist)
                    if sim is not None and sim < self.hsv_hist_intersect_thresh:
                        continue
                    if sim is None and src_bg_mean is not None:
                        dest_bg_mean = self._mean_color_bgr(dest_roi, dest_ring)
                        if dest_bg_mean is not None:
                            if float(np.linalg.norm(dest_bg_mean - src_bg_mean)) > self.semantic_bg_color_thresh:
                                continue

                    # Compose
                    mixed_clone = img.copy()
                    if self.blend_mode == "poisson":
                        try:
                            center_placement = (dst_center_x, dst_center_y)
                            mixed_clone = cv2.seamlessClone(
                                obj_rgb, img, blend_mask, center_placement, cv2.NORMAL_CLONE
                            )
                        except Exception:
                            last_skip = "SKIP: poisson_failed"
                            continue
                    else:
                        alpha = obj_alpha
                        if self.feather_radius > 0:
                            alpha = self._alpha_feather(alpha, self.feather_radius)
                        alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]
                        roi = mixed_clone[y1:y2, x1:x2].astype(np.float32)
                        src = obj_rgb.astype(np.float32)
                        mixed = src * alpha_f + roi * (1.0 - alpha_f)
                        mixed_clone[y1:y2, x1:x2] = mixed.astype(np.uint8)

                    # Global post-processing
                    mixed_clone = self._post_process_composite(mixed_clone)

                    # Tri-map
                    tri_map = np.zeros((h_img, w_img), dtype=np.uint8)
                    tri_map[src_mask_full > 0] = 128
                    tri_map[y1:y2, x1:x2][obj_mask_bin > 0] = 255

                    # Soft target mask (optional additional output)
                    soft = self._make_soft_target((tri_map == 255).astype(np.uint8))
                    soft_name = None
                    if soft is not None:
                        soft_name = f"{image_info['file_name'].replace('.jpg', '')}_soft_target.png"
                        cv2.imwrite(os.path.join(self.output_dir_masks, "soft_targets", soft_name), soft)

                    # Metadata + save
                    out_base = image_info['file_name'].replace('.jpg', '')
                    output_data = {
                        'image_filename': f"{out_base}.png",
                        'mask_filename': f"{out_base}_mask.png",
                        'soft_target_filename': soft_name,
                        'transform_policy': self.transform_policy,
                        'flip': int(do_flip),
                        'blend_mode': self.blend_mode,
                        'scale_factor': round(scale_factor, 3),
                        'rotation_angle': round(rotation_angle, 3),
                        'source_bbox': (src_x, src_y, src_w, src_h),
                        'target_bbox': (dst_x, dst_y, pat_w, pat_h),
                    }
                    cv2.imwrite(os.path.join(self.output_dir_tampered, output_data['image_filename']), mixed_clone)
                    cv2.imwrite(os.path.join(self.output_dir_masks, output_data['mask_filename']), tri_map)
                    return output_data

                last_skip = "SKIP: no_valid_destination"

        return last_skip
