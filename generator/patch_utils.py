"""
Patch-level utility functions for MemoryForger architecture alignment.

Provides patch tokenization, label generation, and image padding
so that the dataset outputs are directly consumable by the
InfoNCE contrastive training pipeline.
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

    Each patch is classified based on majority vote of its pixels:
    - 0 = background patch (clean)
    - 1 = source patch (>50% source pixels, value 128)
    - 2 = target patch (>50% target pixels, value 255)
    - 3 = mixed/boundary (ambiguous — exclude from contrastive loss)

    Parameters
    ----------
    trimap : np.ndarray
        (H, W) uint8 trimap with values {0, 128, 255}.
    patch_size : int
        Patch side length.

    Returns
    -------
    np.ndarray
        (N_h, N_w) uint8 patch label grid.
    """
    h, w = trimap.shape[:2]
    n_h = h // patch_size
    n_w = w // patch_size
    total_pixels = patch_size * patch_size
    labels = np.zeros((n_h, n_w), dtype=np.int8)

    for i in range(n_h):
        for j in range(n_w):
            patch = trimap[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size,
            ]
            src_frac = float((patch == 128).sum()) / total_pixels
            tgt_frac = float((patch == 255).sum()) / total_pixels

            if tgt_frac > 0.5:
                labels[i, j] = 2  # target
            elif src_frac > 0.5:
                labels[i, j] = 1  # source
            elif src_frac + tgt_frac > 0.1:
                labels[i, j] = -1  # mixed boundary — skip in loss
            # else: 0 = clean background

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
    mix_count = int((labels == -1).sum())

    return {
        "total_patches": total,
        "background_patches": bg_count,
        "source_patches": src_count,
        "target_patches": tgt_count,
        "boundary_patches": mix_count,
        "source_fraction": round(src_count / max(total, 1), 4),
        "target_fraction": round(tgt_count / max(total, 1), 4),
    }
