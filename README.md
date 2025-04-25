# 🧠 FOMO25 Challenge Codebase

This repository contains the official baseline code for the FOMO25 Challenge, which investigates the few-shot generalization properties of foundation models for brain MRI data analysis.

Please note: This codebase will be continually refined, so check back occasionally to get the latest updates.

## 🔍 Challenge Overview

The FOMO25 Challenge seeks to advance the field of medical image analysis by evaluating the same pretrained models across multiple downstream tasks. Participants will first pretrain on a large unlabelled dataset and then evaluate their models on three clinical, multi-vendor, and multi-center datasets. For more information on the challenge, please visit the [FOMO25 Challenge website](https://fomo25.github.io).

### ⏱️ Timeline
- April 1: Challenge Opens
- April 7: Code Release
- April 15: Sanity-Check Code Access
- June 15: Validation Leaderboard and Final Submission Pipeline Opens
- August 20: Challenge Submission Deadline
- September 22/28, 2025: Results Release at MICCAI 2025

## 📋 Tasks and Data

This codebase supports three tasks:

1. **Task 1: Infarct Detection** - Binary classification of brain infarcts
2. **Task 2: Meningioma Segmentation** - Binary segmentation of brain meningiomas
3. **Task 3: Brain Age Regression** - Predicting patient age from MRI scans

Data for the challenge includes:
- **Pretraining Data**: 11,187 subjects, 13,900 sessions, 60,529 scans
- **Finetuning Data**: Limited few-shot data (~20-200 cases per task)
- **Evaluation Data**: Unseen data for final assessment

## 🛠️ Requirements

Install required dependencies:

```bash
# Install basic dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For testing
pip install -e ".[test]"

# For all dependencies
pip install -e ".[dev,test]"
```

## Data Preparation

While the data included in this challenge is already preprocessed (co-registered, transposed to RAS orientation and defaced/skull-stripped), to run this code, one needs to further preprocess with the following _highly opinionated preprocessing_ steps.

This "Opinionated Preprocessing" can be done in the following way

### Yucca Preprocess Pretraining Data (required)

For preprocessing the pretraining (FOMO60K) data:

```bash
python src/data/fomo-60k/preprocess.py --in_path=/path/to/raw/pretrain/data --out_path=/path/to/output/preprocessed/data
```

This will:

1. Store each tensor in numpy format for easy loading.
2. Treat each scan as a separate datapoint which can be sampled iid.
3. Crop to the minimum bounding box.
4. Z-normalize on a per-volume level.
5. Resample to isotropic (1mm, 1mm, 1mm) spacing.


### Yucca Preprocess Finetuning Data (required)

For preprocessing the finetuning data for tasks 1-3:

```bash
python src/data/preprocess/run_preprocessing.py --taskid=1 --source_path=/path/to/raw/finetuning/data
```

Replace `--taskid=1` with `--taskid=2` or `--taskid=3` for the other tasks.

This will:

1. Assemble each session into a single 4D tensor and store it as a numpy array for easy loading.
2. Crop to the minimum bounding box.
3. Z-normalize on a per-volume level.
4. Resample to isotropic (1mm, 1mm, 1mm) spacing.


## Pretraining

To pretrain a model using the AMAES (Augmented Masked Autoencoder) framework:

```bash
python src/pretrain.py \
    --save_dir=/path/to/save/models \
    --pretrain_data_dir=/path/to/preprocessed/pretrain/data \
    --model_name=unet_b_lw_dec \
    --patch_size=96 \
    --batch_size=2 \
    --epochs=100 \
    --warmup_epochs=5 \
    --num_workers=64 \
    --augmentation_preset=all
```

Key pretraining parameters:
- `--model_name`: Supported models include `unet_b_lw_dec`, `unet_xl_lw_dec`, etc.
- `--patch_size`: Size of 3D patches (must be divisible by 8)
- `--mask_patch_size`: Size of masking unit for MAE (default is 4)
- `--mask_ratio`: Ratio of patches to mask (default is 0.6)
- `--augmentation_preset`: Choose from `all`, `basic`, or `none`

## Finetuning

To finetune a pretrained model on one of the three tasks:

```bash
python src/finetune.py \
    --data_dir=/path/to/preprocessed/data \
    --save_dir=/path/to/save/finetuned/models \
    --pretrained_weights_path=/path/to/pretrained/checkpoint.pth \
    --model_name=unet_b \
    --patch_size=32 \
    --taskid=1 \
    --batch_size=2 \
    --epochs=500 \
    --train_batches_per_epoch=100 \
    --augmentation_preset=basic
```

Key finetuning parameters:
- `--taskid`: Task ID (1: Infarct Detection, 2: Meningioma Segmentation, 3: Brain Age Regression)
- `--model_name`: Must match the architecture of the pretrained checkpoint
- `--pretrained_weights_path`: Path to the pretrained model checkpoint
- `--augmentation_preset`: Choose from `all`, `basic`, or `none`

## 💾 Model Checkpoints

Pre-trained model checkpoints coming soon...

Note: When finetuning a specific checkpoint, make sure to use the matching model architecture. For example, to finetune `unet_xl_lw_dec_fullaug.pth`, use `--model_name=unet_xl`.

## 💻 Hardware Requirements

The reference implementation was pretrained on 2xH100 GPUs with 80GB of memory. Depending on your hardware, you may need to adjust batch sizes and patch sizes accordingly.

## 📝 Submission Guidelines

Participants can compete in two tracks:
1. **Methods Track** 🧪: Only using the provided FOMO60K dataset for pretraining
2. **Open Track** 🌐: Using any data for pretraining (private data allowed)

For both tracks, additional data is not allowed for finetuning. More details about the submission process can be found on the [FOMO25 Challenge website](https://fomo25.github.io/challenge).

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{munk2024amaes,
  title={AMAES: Augmented Masked Autoencoder Pretraining on Public Brain MRI Data for 3D-Native Segmentation},
  author={Munk, Asbjørn and Ambsdorf, Jakob and Llambias, Sebastian and Nielsen, Mads},
  journal={MICCAI Workshop on Advancing Data Solutions in Medical Imaging AI (ADSMI 2024)},
  year={2024}
}

@article{llambias2024yucca,
  title={Yucca: A deep learning framework for medical image analysis},
  author={Llambias, Sebastian N{\o}rgaard and Machnio, Julia and Munk, Asbj{\o}rn and Ambsdorf, Jakob and Nielsen, Mads and Ghazi, Mostafa Mehdipour},
  journal={arXiv preprint arXiv:2407.19888},
  year={2024}
}
```
