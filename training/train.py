# Adapted from DeepfakeBench; see NOTICE.md and LICENSE.
import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
import torch.distributed as dist
from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR
from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from reproducibility import configure, loader_options, sampler_options

parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument(
    "--detector_path", type=str, required=True, help="path to detector YAML file"
)
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument(
    "--no-save_ckpt", dest="save_ckpt", action="store_false", default=True
)
parser.add_argument(
    "--no-save_feat", dest="save_feat", action="store_false", default=True
)
parser.add_argument("--ddp", action="store_true", default=False)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--task_target",
    type=str,
    default="",
    help="specify the target of current training task",
)
parser.add_argument(
    "--init_weights",
    type=str,
    help="Detector state_dict for fine-tuning; not optimizer resume",
)
args = parser.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError("Training requires a CUDA GPU; CPU is supported for evaluation")
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config["manualSeed"] is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    if config["cuda"]:
        torch.manual_seed(config["manualSeed"])
        torch.cuda.manual_seed_all(config["manualSeed"])


def build_weighted_sampler(train_set, config, logger=None):
    """
    Build a WeightedRandomSampler for imbalanced classification.

    For DeepfakeAbstractBaseDataset, the dataset index is aligned with:
        train_set.data_dict['label']
    """
    if not hasattr(train_set, "data_dict"):
        if logger is not None:
            logger.info("WeightedRandomSampler disabled: train_set has no data_dict.")
        return None
    if "label" not in train_set.data_dict:
        if logger is not None:
            logger.info(
                "WeightedRandomSampler disabled: train_set.data_dict has no 'label' key."
            )
        return None
    numeric_labels = list(train_set.data_dict["label"])
    if len(numeric_labels) == 0:
        if logger is not None:
            logger.info("WeightedRandomSampler disabled: empty training label list.")
        return None
    class_counts = {}
    for lab in numeric_labels:
        class_counts[lab] = class_counts.get(lab, 0) + 1
    if len(class_counts) < 2:
        if logger is not None:
            logger.info(
                f"WeightedRandomSampler disabled: only one class found: {class_counts}"
            )
        return None
    sample_weights = [1.0 / class_counts[lab] for lab in numeric_labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        **sampler_options(config),
    )
    if logger is not None:
        logger.info(f"WeightedRandomSampler enabled. Class counts: {class_counts}")
    return sampler


def prepare_training_data(config, logger=None):
    train_set = DeepfakeAbstractBaseDataset(config=config, mode="train")
    controlled_options = loader_options(config, train_set)
    if config["ddp"]:
        sampler = DistributedSampler(train_set)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            num_workers=int(config["workers"]),
            collate_fn=train_set.collate_fn,
            sampler=sampler,
            **controlled_options,
        )
    else:
        weighted_sampler = None
        if config.get("use_weighted_sampler", False):
            weighted_sampler = build_weighted_sampler(train_set, config, logger=logger)
        if weighted_sampler is not None:
            train_data_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config["train_batchSize"],
                sampler=weighted_sampler,
                shuffle=False,
                num_workers=int(config["workers"]),
                collate_fn=train_set.collate_fn,
                **controlled_options,
            )
        else:
            train_data_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config["train_batchSize"],
                shuffle=True,
                num_workers=int(config["workers"]),
                collate_fn=train_set.collate_fn,
                **controlled_options,
            )
    return train_data_loader


def prepare_testing_data(config):

    def get_test_data_loader(config, test_name, stream):
        config = config.copy()
        config["test_dataset"] = test_name
        test_set = DeepfakeAbstractBaseDataset(config=config, mode="test")
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config["test_batchSize"],
            shuffle=False,
            num_workers=int(config["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=test_name == "DeepFakeDetection",
            **loader_options(config, test_set, stream=stream),
        )
        return test_data_loader

    test_data_loaders = {}
    for stream, one_test_name in enumerate(config["test_dataset"], 1):
        test_data_loaders[one_test_name] = get_test_data_loader(
            config, one_test_name, stream
        )
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config["optimizer"]["type"]
    if opt_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
        )
        return optimizer
    elif opt_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"][opt_name]["lr"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
            betas=(
                config["optimizer"][opt_name]["beta1"],
                config["optimizer"][opt_name]["beta2"],
            ),
            eps=config["optimizer"][opt_name]["eps"],
            amsgrad=config["optimizer"][opt_name]["amsgrad"],
        )
        return optimizer
    elif opt_name == "sam":
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
        )
    else:
        raise NotImplementedError(
            "Optimizer {} is not implemented".format(config["optimizer"])
        )
    return optimizer


def choose_scheduler(config, optimizer):
    if config["lr_scheduler"] is None:
        return None
    elif config["lr_scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config["lr_step"], gamma=config["lr_gamma"]
        )
        return scheduler
    elif config["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["lr_T_max"], eta_min=config["lr_eta_min"]
        )
        return scheduler
    elif config["lr_scheduler"] == "linear":
        scheduler = LinearDecayLR(
            optimizer, config["nEpochs"], int(config["nEpochs"] / 4)
        )
        return scheduler
    else:
        raise NotImplementedError(
            "Scheduler {} is not implemented".format(config["lr_scheduler"])
        )


def choose_metric(config):
    metric_scoring = config["metric_scoring"]
    if metric_scoring not in ["eer", "auc", "acc", "ap"]:
        raise NotImplementedError("metric {} is not implemented".format(metric_scoring))
    return metric_scoring


def main():
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
    with open("./training/config/train_config.yaml", "r") as f:
        config2 = yaml.safe_load(f)
    if "label_dict" in config:
        config2["label_dict"] = config["label_dict"]
    config.update(config2)
    from runtime import apply_runtime

    config = apply_runtime(config)
    config["local_rank"] = args.local_rank
    if config["dry_run"]:
        config["nEpochs"] = 0
        config["save_feat"] = False
    if args.train_dataset:
        config["train_dataset"] = args.train_dataset
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    config["save_ckpt"] = args.save_ckpt
    config["save_feat"] = args.save_feat
    timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task_name = args.task_target if args.task_target else config.get("task_target", "")
    task_str = f"_{task_name}" if task_name else ""
    logger_path = os.path.join(
        config["log_dir"], f"{config['model_name']}{task_str}_{timenow}"
    )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, "training.log"))
    logger.info("Save log to {}".format(logger_path))
    config["ddp"] = args.ddp
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)
    init_seed(config)
    configure(config)
    if config["cudnn"]:
        cudnn.benchmark = True
    if config["ddp"]:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))
    train_data_loader = prepare_training_data(config, logger=logger)
    test_data_loaders = prepare_testing_data(config)
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)
    if args.init_weights:
        model.load_state_dict(
            torch.load(args.init_weights, map_location="cpu", weights_only=True),
            strict=True,
        )
        logger.info(
            "Fine-tuning from %s; optimizer is initialized afresh", args.init_weights
        )
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)
    trainer = Trainer(
        config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow
    )
    for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!"
            )
    logger.info(
        "Stop Training on best Testing metric {}".format(
            parse_metric_for_print(best_metric)
        )
    )
    if scheduler is not None:
        scheduler.step()
    for writer in trainer.writers.values():
        writer.close()


if __name__ == "__main__":
    main()
