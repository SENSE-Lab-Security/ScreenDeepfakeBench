"""DeepfakeBench-derived loader preserving the paper's OpenCV BGR inputs."""

from copy import deepcopy
import json
from pathlib import Path
import random
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F
from .albu import IsotropicResize


def frame_key(path):
    stem = Path(path).stem
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else stem


class DeepfakeAbstractBaseDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config, self.mode = config, mode
        self.root = Path(config["rgb_dir"])
        if config.get("with_mask") or config.get("with_landmark") or config.get("lmdb"):
            raise ValueError(
                "Released configurations use frames without masks, landmarks or LMDB"
            )
        names = config["train_dataset"] if mode == "train" else [config["test_dataset"]]
        self.image_list, self.label_list = [], []
        for name in names:
            obj = json.loads(
                (Path(config["dataset_json_folder"]) / (name + ".json")).read_text()
            )[name]
            pairs = []
            for label, groups in obj.items():
                for video in groups[mode].values():
                    frames = sorted(video["frames"], key=frame_key)[
                        : config["frame_num"][mode]
                    ]
                    pairs.extend(
                        (config["label_dict"][video.get("label", label)], p)
                        for p in frames
                    )
            random.shuffle(pairs)
            self.label_list.extend(y for y, _ in pairs)
            self.image_list.extend(p for _, p in pairs)
        if not self.image_list:
            raise ValueError("The selected data index contains no frames")
        self.data_dict = {"image": self.image_list, "label": self.label_list}
        self.transform = self.init_data_aug_method()
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=config["mean"], std=config["std"])

    def init_data_aug_method(self):
        if self.mode == "test":
            return A.Compose([])
        c, a = self.config, self.config["data_aug"]
        resize = IsotropicResize(
            max_side=c["resolution"],
            interpolation_down=cv2.INTER_AREA,
            interpolation_up=cv2.INTER_LINEAR,
            p=1.0,
        )
        mode = c.get("recapture_aug_mode", "baseline")
        if mode == "coc":
            return A.Compose(
                [
                    A.Perspective(scale=(0.05, 0.10), keep_size=True, p=c["geo_prob"]),
                    A.Downscale(scale_range=(0.6, 0.9), p=c["dist_prob"]),
                    resize,
                    A.RandomBrightnessContrast(
                        brightness_limit=a["brightness_limit"],
                        contrast_limit=a["contrast_limit"],
                        p=c["brightness_contrast_prob"],
                    ),
                    A.HueSaturationValue(p=c["hsv_prob"]),
                    A.FancyPCA(p=c["pca_prob"]),
                    A.ImageCompression(quality_range=(20, 70), p=c["comp_strong_prob"]),
                ]
            )
        if mode != "baseline":
            raise ValueError("Unsupported augmentation mode: " + mode)
        return A.Compose(
            [
                A.HorizontalFlip(p=a["flip_prob"]),
                A.Rotate(limit=a["rotate_limit"], p=a["rotate_prob"]),
                A.GaussianBlur(blur_limit=a["blur_limit"], p=a["blur_prob"]),
                resize,
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=a["brightness_limit"],
                            contrast_limit=a["contrast_limit"],
                            p=1.0,
                        ),
                        A.FancyPCA(p=1.0),
                        A.HueSaturationValue(p=1.0),
                    ],
                    p=0.5,
                ),
                A.ImageCompression(
                    quality_range=(a["quality_lower"], a["quality_upper"]), p=0.5
                ),
            ]
        )

    def load_rgb(self, path):
        image = cv2.imread(str(self.root / path))
        if image is None:
            raise FileNotFoundError(self.root / path)
        return image

    def __getitem__(self, index, no_norm=False):
        image = self.load_rgb(self.image_list[index])
        if self.mode == "train" and self.config["use_data_augmentation"]:
            image = self.transform(image=image)["image"]
        else:
            image = deepcopy(image)
        if not no_norm:
            image = self.normalize(self.to_tensor(image))
        return image, self.label_list[index], None, None

    @staticmethod
    def collate_fn(batch):
        images, labels, _, _ = zip(*batch)
        resized = [
            (
                F.interpolate(
                    x.unsqueeze(0),
                    size=(299, 299),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if x.shape[-2:] != (299, 299)
                else x
            )
            for x in images
        ]
        return {
            "image": torch.stack(resized),
            "label": torch.LongTensor(labels),
            "landmark": None,
            "mask": None,
        }

    def __len__(self):
        return len(self.image_list)
