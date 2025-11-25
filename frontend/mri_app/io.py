import io as pyio
import tempfile
from nibabel.loadsave import load
import numpy as np
from typing import Tuple

def load_nifti_uploaded(uploaded_file) -> Tuple[np.ndarray, tuple[float, float, float], str]:
    """
    Load a Streamlit UploadedFile into image [C, H, W, D], spacing, and a case_id.

    returns:
            image: MRI volume float32
            spacing voxel spacing mm from header (tuple of floats)
            case_id: filename without extension
    """
    case_id = uploaded_file.name[:-7]
    # 2. Write uploaded bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=True, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
        img = load(temp_path)
        image = img.get_fdata().astype("float32")  # pyright: ignore[reportAttributeAccessIssue] # [H,W,D] or [C,H,W,D] depending on BRATS version
    # raw = uploaded_file.read()
    # fileobj = pyio.BytesIO(raw)

    # if data.ndim == 3:
        # assume single channel → add C dimension
        # data = data[None, ...]  # [1,H,W,D]

    # Reorder to [C,D,H,W]
    image = np.moveaxis(image, -1, 0)  # [C,D,H,W]

    hdr = img.header
    spacing = tuple(float(s) for s in hdr.get_zooms()[0:3])  # pyright: ignore[reportAttributeAccessIssue] # (sx, sy, sz)

    case_id = uploaded_file.name.rsplit(".", 2)[0]  # e.g. "BRATS_001"
    return image, spacing, case_id # pyright: ignore[reportReturnType]