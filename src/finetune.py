#!/usr/bin/env python

import argparse
import os
import logging
import torch
import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

from models.supervised_base import BaseSupervisedModel
from augmentations.finetune_augmentation_presets import (
    get_finetune_augmentation_params,
)
from utils.utils import (
    SimplePathConfig,
    setup_seed,
    find_checkpoint,
    load_pretrained_weights,
)

from batchgenerators.utilities.file_and_folder_operations import (
    maybe_mkdir_p as ensure_dir_exists,
)

from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset

from yucca.pipeline.configuration.split_data import get_split_config
from yucca.pipeline.configuration.configure_paths import detect_version
from data.dataset import FOMODataset
from data.task_configs import task1_config, task2_config, task3_config


def get_task_config(taskid):
    if taskid == 1:
        task_cfg = task1_config
    elif taskid == 2:
        task_cfg = task2_config
    elif taskid == 3:
        task_cfg = task3_config
    else:
        raise ValueError(f"Unknown taskid: {taskid}. Supported IDs are 1, 2, and 3")

    return task_cfg


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory",
        default="./data/preprocessed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save models and results",
        default="./data/models",
    )
    parser.add_argument(
        "--pretrained_weights_path", type=str, help="Ckpt to finetune", default=None
    )
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_b",
        help="Model name defined in models.networks (unet_b, unet_xl, etc.)",
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    # Hardware configuration
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fast_dev_run", action="store_true")
    # Experiment tracking
    parser.add_argument("--new_version", action="store_true")
    parser.add_argument(
        "--augmentation_preset",
        type=str,
        choices=["all", "basic", "none"],
        default="basic",
    )
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_batches_per_epoch", type=int, default=100)
    # Task Configuration
    parser.add_argument(
        "--taskid",
        type=int,
        required=True,
        help="Task ID (1: FOMO1 classification, 2: FOMO2 classification, 3: FOMO3 regression)",
    )
    # Split Configuration
    parser.add_argument("--split_method", type=str, default="simple_train_val_split")
    parser.add_argument("--split_param", type=str, help="Split parameter", default=0.2)
    parser.add_argument(
        "--split_idx", type=int, default=0, help="Index of the split to use for kfold"
    )
    parser.add_argument(
        "--experiment", type=str, default="experiment", help="name of experiment"
    )
    args = parser.parse_args()

    assert (
        args.patch_size % 8 == 0
    ), f"Patch size must be divisible by 8, got {args.patch_size}"

    # Set up task configuration
    task_cfg = get_task_config(args.taskid)
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    num_classes = task_cfg["num_classes"]
    modalities = len(task_cfg["modalities"])
    labels = task_cfg["labels"]

    run_type = "from_scratch" if args.pretrained_weights_path is None else "finetune"
    experiment_name = f"{run_type}_{args.experiment}_{args.taskid}"

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print(f"Task type: {task_type}")
    print("ARGS:", args)

    # Set up directory structure
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, task_name)

    # Path where logs, checkpoints etc is stored
    save_dir = os.path.join(args.save_dir, task_name, args.model_name)

    # Handle versioning for experiment tracking
    continue_from_most_recent = not args.new_version
    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = os.path.join(save_dir, f"version_{version}")
    ensure_dir_exists(version_dir)

    # Create dataset splits
    if args.split_method == "kfold":
        split_param = int(args.split_param)
    elif args.split_method == "simple_train_val_split":
        split_param = float(args.split_param)
    else:
        split_param = args.split_param

    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    splits_config = get_split_config(
        method=args.split_method,
        param=split_param,
        path_config=path_config,
    )

    # Set up seed for reproducability
    seed = setup_seed(continue_from_most_recent)
    # Look for existing checkpoint if continuing training
    ckpt_path = find_checkpoint(version_dir, continue_from_most_recent)

    # Calculate training metrics
    effective_batch_size = args.num_devices * args.batch_size
    train_dataset_size = len(splits_config.train(args.split_idx))
    val_dataset_size = len(splits_config.val(args.split_idx))
    max_iterations = int(args.epochs * args.train_batches_per_epoch)

    # the config contains all the parameters needed for training and is used by lightning module, data module, and trainer
    config = {
        # Task information
        "task": task_name,
        "task_id": args.taskid,
        "task_type": task_type,
        "experiment": experiment_name,
        "model_name": args.model_name,
        "model_dimensions": "3D",
        "run_type": run_type,
        # Split configuration
        "split_method": args.split_method,
        "split_param": split_param,
        "split_idx": args.split_idx,
        # Directories
        "save_dir": save_dir,
        "train_data_dir": train_data_dir,
        "version_dir": version_dir,
        "version": version,
        # Checkpoint
        "ckpt_path": ckpt_path,
        "pretrained_weights_path": args.pretrained_weights_path,
        # Reproducibility
        "seed": seed,
        # Dataset properties
        "num_classes": num_classes,
        "num_modalities": modalities,
        "image_extension": ".npy",
        "allow_missing_modalities": False,
        "labels": labels,
        # Training parameters
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patch_size": (args.patch_size,) * 3,
        "precision": args.precision,
        "augmentation_preset": args.augmentation_preset,
        "epochs": args.epochs,
        "train_batches_per_epoch": args.train_batches_per_epoch,
        "effective_batch_size": effective_batch_size,
        # Dataset metrics
        "train_dataset_size": train_dataset_size,
        "val_dataset_size": val_dataset_size,
        "max_iterations": max_iterations,
        # Hardware settings
        "num_devices": args.num_devices,
        "num_workers": args.num_workers,
        # Model compilation
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        # Trainer specific params
        "fast_dev_run": args.fast_dev_run,
    }

    # Create checkpoint callback for saving models
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=1,
        filename="last",
        enable_version_counter=False,
    )
    callbacks = [checkpoint_callback]

    # Create logger for metrics
    yucca_logger = YuccaLogger(
        save_dir=save_dir,
        version=version,
        steps_per_epoch=args.train_batches_per_epoch,
    )

    # Create wandb logger for Lightning
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="fomo-finetuning",
        name=f"{config['experiment']}_version_{config['version']}",
        log_model=True,
    )

    # Set up loggers
    loggers = [yucca_logger, wandb_logger]

    # Configure augmentations based on preset
    aug_params = get_finetune_augmentation_params(args.augmentation_preset)
    # we use the cls augmentatoin preset for regression
    tt_preset = "classification" if task_type == "regression" else task_type
    augmenter = YuccaAugmentationComposer(
        patch_size=config["patch_size"],
        task_type_preset=tt_preset,
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    # Create the data module that handles loading and batching
    data_module = YuccaDataModule(
        train_dataset_class=(
            YuccaTrainDataset if task_type == "segmentation" else FOMODataset
        ),
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
        train_data_dir=config["train_data_dir"],
        image_extension=config["image_extension"],
        task_type=config["task_type"],
        splits_config=splits_config,
        split_idx=config["split_idx"],
        num_workers=args.num_workers,
        val_sampler=None,
    )

    # Print dataset information
    print("Train dataset: ", data_module.splits_config.train(config["split_idx"]))
    print("Val dataset: ", data_module.splits_config.val(config["split_idx"]))
    print("Run type: ", run_type)
    print(
        f"Starting training with {max_iterations} max iterations over {args.epochs} epochs "
        f"with train dataset of size {train_dataset_size} datapoints and val dataset of size {val_dataset_size} "
        f"and effective batch size of {effective_batch_size}"
    )

    # Create model and trainer
    model = BaseSupervisedModel.create(
        task_type=task_type,
        config=config,
        learning_rate=args.learning_rate,
        do_compile=args.compile,
        compile_mode="default" if args.compile_mode is None else args.compile_mode,
    )

    # Create Lightning trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        num_nodes=1,
        devices=args.num_devices,
        default_root_dir=save_dir,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches_per_epoch,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
    )

    # Load pretrained weights if finetuning
    if run_type == "finetune":
        print("Transferring weights for finetuning")
        print(f"Checkpoint path: {ckpt_path}")
        assert ckpt_path is None, (
            "Error: You're attempting to load pretrained weights while "
            "simultaneously continuing from a checkpoint. This creates "
            "conflicting weight sources. Use either --pretrained_weights_path "
            "for finetuning OR continue training without the --new_version flag, "
            "but not both."
        )
        # Load and adjust weights from pretrained model
        state_dict = load_pretrained_weights(args.pretrained_weights_path, args.compile)

        # Transfer weights to new model
        num_successful_weights_transferred = model.load_state_dict(
            state_dict=state_dict, strict=False
        )
        assert (
            num_successful_weights_transferred > 0
        ), "No weights were successfully transferred"
    else:
        print("Training from scratch, no weights will be transferred")

    # Start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path="last")
    wandb.finish()


if __name__ == "__main__":
    main()
