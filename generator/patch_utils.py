"""
Patch-level utility functions for MemoryForger architecture alignment.

Provides patch tokenization, label generation, and image padding
so that the dataset outputs are directly consumable by the
InfoNCE contrastive training pipeline.

Design note (F3 — Strict Purity):
    trimap_to_patch_labels now uses a 100%-purity rule.
    Any patch that is not completely homogeneous (all background,
    all source, or all target) is labeled -1 (ignore).  This
    prevents the CNN from using the jagged copy-move boundary as a
    shortcut in the InfoNCE loss — the model is forced to inspect
    deep interior statistics instead of high-frequency edge cues.
"""

import numpy as np


def pad_to_patch_grid(img, patch_size=16):
    """
    Pad an image so both dimensions are divisible by patch_size.

    Uses reflect padding to avoid introducing black borders that
    would confuse the encoder.

    Parameters
    ----------
    img : np.ndarray
        Image array, either (H, W, C) or (H, W).
    patch_size : int
        Patch side length (default 16 for ViT-style tokenization).

    Returns
    -------
    np.ndarray
        Padded image with dimensions divisible by patch_size.
    """
    h, w = img.shape[:2]
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size

    if new_h == h and new_w == w:
        return img

    pad_h = new_h - h
    pad_w = new_w - w

    if img.ndim == 3:
        padded = np.pad(
            img,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )
    else:
        padded = np.pad(
            img,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )
    return padded


def trimap_to_patch_labels(trimap, patch_size=16):
    """
    Convert a pixel-level trimap to patch-level labels.

    STRICT 100%-PURITY RULE (F3):
    A patch is labeled only if *every* pixel in the patch belongs
    to the same class.  Mixed patches (any combination of values)
    receive label -1 and are excluded from the InfoNCE contrastive
    loss.  This eliminates the boundary-cheat shortcut for the CNN.

    Label mapping:
        -1 = ignore  (boundary / mixed — excluded from loss)
         0 = background  (100% value-0 pixels)
         1 = source       (100% value-128 pixels)
         2 = target       (100% value-255 pixels)

    Why 100% purity instead of majority vote?
    -----------------------------------------
    A patch containing even ONE boundary pixel carries the jagged
    copy-move edge as a high-frequency feature.  The InfoNCE loss
    will exploit this shortcut because it has a lower gradient cost
    than learning the color/texture prior.  By making the threshold
    binary (all-or-nothing) we ensure the model only sees clean,
    uncontaminated interior patches during training.

    Parameters
    ----------
    trimap : np.ndarray
        (H, W) uint8 trimap with values {0, 128, 255}.
    patch_size : int
        Patch side length.

    Returns
    -------
    np.ndarray
        (N_h, N_w) int8 patch label grid.
        Values: -1 (ignore), 0 (bg), 1 (source), 2 (target).
    """
    h, w = trimap.shape[:2]
    n_h = h // patch_size
    n_w = w // patch_size
    total_pixels = patch_size * patch_size

    # Use int8 to accommodate -1 (ignore) label
    labels = np.full((n_h, n_w), fill_value=-1, dtype=np.int8)

    for i in range(n_h):
        for j in range(n_w):
            patch = trimap[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size,
            ]

            bg_count = int((patch == 0).sum())
            src_count = int((patch == 128).sum())
            tgt_count = int((patch == 255).sum())

            # Label only perfectly homogeneous patches
            if bg_count == total_pixels:
                labels[i, j] = 0   # clean background
            elif src_count == total_pixels:
                labels[i, j] = 1   # pure source interior
            elif tgt_count == total_pixels:
                labels[i, j] = 2   # pure target interior
            # else: -1 (ignore) — boundary or mixed patch

    return labels


def save_patch_labels(labels, output_path):
    """
    Save patch-level labels as a compressed numpy file.

    Parameters
    ----------
    labels : np.ndarray
        (N_h, N_w) patch label grid.
    output_path : str
        Path to save the .npy file.
    """
    np.save(output_path, labels)


def load_patch_labels(path):
    """Load patch-level labels from a .npy file."""
    return np.load(path)


def get_patch_statistics(labels):
    """
    Compute statistics from a patch label grid.

    Returns
    -------
    dict
        Counts and fractions for each label type.
    """
    total = labels.size
    bg_count = int((labels == 0).sum())
    src_count = int((labels == 1).sum())
    tgt_count = int((labels == 2).sum())
    # -1 encodes as 255 in uint8 but we use int8 here directly
    mix_count = int((labels == -1).sum())

    return {
        "total_patches": total,
        "background_patches": bg_count,
        "source_patches": src_count,
        "target_patches": tgt_count,
        "boundary_patches": mix_count,
        "source_fraction": round(src_count / max(total, 1), 4),
        "target_fraction": round(tgt_count / max(total, 1), 4),
        "ignore_fraction": round(mix_count / max(total, 1), 4),
    }
