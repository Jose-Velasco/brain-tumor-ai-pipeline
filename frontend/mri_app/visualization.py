import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


LABEL_COLORS = {
    0: (0, 0, 0, 0.0),      # background transparent
    1: (0.1, 0.8, 0.1, 0.5),  # edema = green
    2: (1.0, 1.0, 0.0, 0.5),  # non-enh = yellow
    3: (1.0, 0.0, 0.0, 0.5),  # enhancing = red
}

def make_overlay_figure(
    image: np.ndarray,
    mask: np.ndarray,
    slice_idx: int,
    modality_idx: int = 0,
    figsize=(6, 6),
    alpha=0.4, # opacity of overlay
):
    """
    Make a slice visualization with segmentation overlay.

    image:   [C, H, W, D]
    mask:    [H, W, D]
    slice_idx: depth index to visualize
    modality_idx: which MRI channel to display
    """

    # Select channel & slice
    img_slice = image[modality_idx, :, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Show MRI slice
    ax.imshow(img_slice, cmap="gray")

    cmap = ListedColormap([LABEL_COLORS[i] for i in range(4)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    # Colored overlay
    ax.imshow(mask_slice, cmap=cmap, alpha=alpha, norm=norm)

    legend_patches = [
        Patch(color=LABEL_COLORS[1], label="Edema"),
        Patch(color=LABEL_COLORS[2], label="Non-enhancing tumor"),
        Patch(color=LABEL_COLORS[3], label="Enhancing tumor"),
    ]

    ax.legend(handles=legend_patches, loc="lower left")
    ax.axis("off")
    return fig