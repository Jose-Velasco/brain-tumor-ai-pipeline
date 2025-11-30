from monai.transforms.compose import Compose
# from monai.transforms.io.dictionary import LoadImaged
# from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
# from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.intensity.array import NormalizeIntensity

def get_standard_student_teacher_transform() -> Compose:
    return Compose([
        # LoadImaged(keys=["image", "label"]), # not needed as image is already loaded
        # EnsureChannelFirstd(keys=["image", "label"]), # not needed as image has already been properly order in the frontend (maybe move it to backend instead?)
        # NormalizeIntensityd(),
        NormalizeIntensity(),
        # ToTensord(keys=["image", "label"]), # not needed as the imag is already a Tensor
    ])