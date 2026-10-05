"""Resize operator retained from the paper implementation (DeepfakeBench)."""

import cv2
from albumentations import DualTransform


def isotropically_resize_image(
    img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC
):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h, w = h * scale, size
    else:
        scale = size / h
        w, h = w * scale, size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    return cv2.resize(img, (int(w), int(h)), interpolation=interpolation)


class IsotropicResize(DualTransform):
    def __init__(
        self,
        max_side,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC,
        always_apply=False,
        p=1,
    ):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(
        self,
        img,
        interpolation_down=cv2.INTER_AREA,
        interpolation_up=cv2.INTER_CUBIC,
        **params
    ):
        return isotropically_resize_image(
            img, self.max_side, interpolation_down, interpolation_up
        )

    def apply_to_mask(self, img, **params):
        return self.apply(
            img,
            interpolation_down=cv2.INTER_NEAREST,
            interpolation_up=cv2.INTER_NEAREST,
            **params
        )

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")
