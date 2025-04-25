import lightning as pl
from torchvision.transforms import Compose
import logging
import torch
from typing import Literal, Optional, Tuple
from torch.utils.data import DataLoader, Sampler
from yucca.pipeline.configuration.split_data import SplitConfig
from yucca.functional.array_operations.matrix_ops import get_max_rotated_size
from yucca.modules.data.augmentation.transforms.Spatial import Spatial

from data.dataset import PretrainDataset


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        splits_config: SplitConfig,
        split_idx: int,
        train_data_dir: str,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        composed_train_transforms: Optional[Compose] = None,
        composed_val_transforms: Optional[Compose] = None,
    ):
        super().__init__()

        # extract parameters
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.split_idx = split_idx
        self.splits_config = splits_config
        self.train_data_dir = train_data_dir

        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.pre_aug_patch_size = (
            get_max_rotated_size(patch_size)
            if augmentations_include_spatial(composed_train_transforms)
            else None
        )
        assert self.pre_aug_patch_size is None or isinstance(
            self.pre_aug_patch_size, tuple
        )

        self.num_workers = (
            max(0, int(torch.get_num_threads() - 1))
            if num_workers is None
            else num_workers
        )
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        assert stage == "fit"

        # Assign train/val datasets for use in dataloaders
        assert self.train_data_dir is not None
        assert self.split_idx is not None
        assert self.splits_config is not None

        self.train_samples = self.splits_config.train(self.split_idx)
        self.val_samples = self.splits_config.val(self.split_idx)

        self.train_dataset = PretrainDataset(
            self.train_samples,
            data_dir=self.train_data_dir,
            composed_transforms=self.composed_train_transforms,
            pre_aug_patch_size=self.pre_aug_patch_size,  # type: ignore
            patch_size=self.patch_size,
        )

        self.val_dataset = PretrainDataset(
            self.val_samples,
            data_dir=self.train_data_dir,
            composed_transforms=self.composed_val_transforms,
            patch_size=self.patch_size,
        )

    def train_dataloader(self):
        logging.info(f"Starting training with data from: {self.train_data_dir}")
        sampler = (
            self.train_sampler(self.train_dataset)
            if self.train_sampler is not None
            else None
        )

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self):
        sampler = (
            self.val_sampler(self.val_dataset) if self.val_sampler is not None else None
        )

        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=False,
            sampler=sampler,
        )


def augmentations_include_spatial(augmentations):
    if augmentations is None:
        return False

    for augmentation in augmentations.transforms:
        if isinstance(augmentation, Spatial):
            return True

    return False
