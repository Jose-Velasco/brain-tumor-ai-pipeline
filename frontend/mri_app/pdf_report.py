import numpy as np
from typing import Any
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# from earlier: compute_intensity_stats, compute_tumor_stats, compute_bbox_and_com, render_slice_overlays, create_pdf_report
# MODALITY_NAMES = ["FLAIR", "T1", "T1ce", "T2"] 
MODALITY_NAMES = ["FLAIR", "T1w", "t1gd", "T2w"] 

# Map label ids to names used in your label mask
LABEL_MAP = {
    0: "background", 
    1: "edema",
    2: "non-enhancing tumor",
    3: "enhancing tumour"
}

def compute_intensity_stats(image: np.ndarray):
    """
    image: [C, H, W, D]
    Returns dict[modality_name] -> stats.
    """
    C = image.shape[0]
    stats: dict[str, dict] = {}

    for c in range(C):
        name = MODALITY_NAMES[c] if c < len(MODALITY_NAMES) else f"channel_{c}"
        x = image[c].astype(np.float32)  # [H, W, D]

        x_nonzero = x[x != 0]  # avoid background skew
        if x_nonzero.size == 0:
            stats[name] = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "pct_zero": 100.0,
            }
            continue

        stats[name] = {
            "min": float(x_nonzero.min()),
            "max": float(x_nonzero.max()),
            "mean": float(x_nonzero.mean()),
            "std": float(x_nonzero.std()),
            "pct_zero": float((x == 0).mean() * 100.0),
        }

    return stats

def compute_voxel_volume_ml(spacing):
    sx, sy, sz = spacing  # in mm
    voxel_volume_mm3 = sx * sy * sz
    return voxel_volume_mm3 / 1000.0  # convert to ml (cm³)

def compute_tumor_stats(label: np.ndarray, spacing, brain_mask: np.ndarray | None = None):
    """
    label: [H, W, D] int
    spacing: (sx, sy, sz) in mm
    brain_mask: optional [H, W, D] bool/int; if None, caller should pass a real brain mask.
    """
    voxel_volume_ml = compute_voxel_volume_ml(spacing)

    if brain_mask is None:
        # Fallback: treat non-background as "brain region".
        # Better: pass an explicit brain mask from the image (see below).
        brain_mask = label > 0

    brain_voxels = int(brain_mask.sum())
    brain_volume_ml = brain_voxels * voxel_volume_ml

    class_stats: dict[str, dict] = {}
    total_tumor_voxels = 0

    for cls_id, cls_name in LABEL_MAP.items():
        cls_mask = label == cls_id
        n_voxels = int(cls_mask.sum())

        # always store class stats
        class_stats[cls_name] = {
            "voxel_count": n_voxels,
            "volume_ml": n_voxels * voxel_volume_ml,
        }

        # BUT: only add non-background to tumor total
        if cls_id != 0:
            total_tumor_voxels += n_voxels

    total_tumor_volume_ml = total_tumor_voxels * voxel_volume_ml
    percent_brain_affected = (
        (total_tumor_voxels / brain_voxels) * 100.0 if brain_voxels > 0 else 0.0
    )

    return {
        "classes": class_stats,
        "total_tumor_volume_ml": total_tumor_volume_ml,
        "percent_brain_affected": percent_brain_affected,
        "brain_voxel_count": brain_voxels,
        "brain_volume_ml": brain_volume_ml,
    }

def compute_bbox_and_com(label: np.ndarray):
    """
    label: [H, W, D]
    Returns: dict[class_name] -> {
        present: bool,
        bbox: [z_min,z_max,y_min,y_max,x_min,x_max],
        center_of_mass_vox: [z,y,x]
    }
    """
    results: dict[str, dict] = {}

    for cls_id, cls_name in LABEL_MAP.items():
        mask = label == cls_id
        if not mask.any():
            results[cls_name] = {
                "present": False,
                "bbox": None,
                "center_of_mass_vox": None,
            }
            continue

        coords = np.argwhere(mask)  # [N, 3] with (y, x, z)
        ymin, xmin, zmin = coords.min(axis=0)
        ymax, xmax, zmax = coords.max(axis=0)

        com_y, com_x, com_z = center_of_mass(mask.astype(np.float32))  # (y, x, z)

        results[cls_name] = {
            "present": True,
            "bbox": [int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)],
            "center_of_mass_vox": [float(com_z), float(com_y), float(com_x)],
        }

    return results

def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    x = slice_2d.astype(np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def _make_overlay(base: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """base: [H,W] normalized; mask: [H,W] int."""
    base_rgb = np.stack([base, base, base], axis=-1)
    overlay = base_rgb.copy()

    tumor = mask > 0
    overlay[tumor, 0] = 1.0
    overlay[tumor, 1] *= (1.0 - alpha)
    overlay[tumor, 2] *= (1.0 - alpha)
    return overlay


def render_slice_overlays(
    image: np.ndarray,
    mask: np.ndarray,
    modality_idx: int = 0,
) -> dict[str, bytes]:
    """
    image: [C, H, W, D], mask: [H, W, D]
    Returns dict: view_name -> PNG bytes (axial, coronal, sagittal).
    """
    vol = image[modality_idx]  # [H, W, D]
    H, W, D = vol.shape

    axial_idx    = D // 2
    coronal_idx  = H // 2
    sagittal_idx = W // 2

    slices = {
        # Axial: plane perpendicular to z (depth), shape [H,W]
        "axial_mid": (
            vol[:, :, axial_idx],
            mask[:, :, axial_idx],
        ),
        # Coronal: plane perpendicular to y, shape [W,D] (still 2D)
        "coronal_mid": (
            vol[coronal_idx, :, :],
            mask[coronal_idx, :, :],
        ),
        # Sagittal: plane perpendicular to x, shape [H,D]
        "sagittal_mid": (
            vol[:, sagittal_idx, :],
            mask[:, sagittal_idx, :],
        ),
    }

    png_dict: dict[str, bytes] = {}
    for name, (img_slice, mask_slice) in slices.items():
        img_norm = _normalize_slice(img_slice)
        overlay = _make_overlay(img_norm, mask_slice)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(overlay)
        ax.set_title(name.replace("_", " ").title())
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        png_dict[name] = buf.getvalue()

    return png_dict

def png_bytes_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")

def build_case_report(case_id: str, image: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, Any]:
    """Builds the JSON-like case_report consumed by the LLM and used in the PDF."""
    # use your earlier helper functions
    intensity_stats = compute_intensity_stats(image)

    # Approximate brain mask: any voxel where at least one modality is non-zero
    # image: [C, H, W, D] -> [H, W, D]
    brain_mask = np.any(image != 0, axis=0)
    tumor_stats = compute_tumor_stats(mask, spacing, brain_mask=brain_mask)
    
    geometry = compute_bbox_and_com(mask)

    return {
        "case_id": case_id,
        "image_shape": list(image.shape),
        "spacing_mm": list(spacing),
        "intensity_stats": intensity_stats,
        "tumor_stats": tumor_stats,
        "geometry": geometry,
    }