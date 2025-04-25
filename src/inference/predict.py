#!/usr/bin/env python

import os
import torch
import numpy as np
import nibabel as nib
from typing import List, Dict, Any
from models.supervised_seg import SupervisedSegModel
from models.supervised_cls import SupervisedClsModel
from models.supervised_reg import SupervisedRegModel
from data.task_configs import task1_config, task2_config, task3_config

from yucca.functional.preprocessing import (
    preprocess_case_for_inference,
    reverse_preprocessing,
)


def get_task_config(taskid):
    """Get task configuration based on task ID."""
    if taskid == 1:
        task_cfg = task1_config
    elif taskid == 2:
        task_cfg = task2_config
    elif taskid == 3:
        task_cfg = task3_config
    else:
        raise ValueError(f"Unknown taskid: {taskid}. Supported IDs are 1, 2, and 3")

    return task_cfg


def load_modalities(modality_paths: List[str]) -> List[nib.Nifti1Image]:
    """Load modality images from provided paths."""
    images = []
    for path in modality_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modality file not found: {path}")
        try:
            img = nib.load(path)
            images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {str(e)}")

    return images


def save_prediction(
    prediction: np.ndarray, reference_img: nib.Nifti1Image, output_path: str
):
    """Save prediction as a NIfTI file using affine from reference image."""
    # Create a new NIfTI image with the prediction data and reference affine
    pred_nifti = nib.Nifti1Image(prediction, reference_img.affine)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Make sure output path has .nii.gz extension
    if not output_path.endswith((".nii", ".nii.gz")):
        output_path = output_path + ".nii.gz"

    # Save the prediction
    nib.save(pred_nifti, output_path)


def predict_from_config(
    modality_paths: List[str],
    output_path: str,
    predict_config: Dict[str, Any],
):
    """
    Run inference on input modality images using a task-specific configuration.

    Args:
        modality_paths: Paths to input modality images
        output_path: Path where prediction will be saved
        predict_config: Dictionary containing all the configuration parameters for prediction

    Returns:
        str: Path to saved prediction
    """
    # Load input images
    images = load_modalities(modality_paths)

    # Extract configuration parameters
    task_type = predict_config["task_type"]
    crop_to_nonzero = predict_config["crop_to_nonzero"]
    norm_op = predict_config["norm_op"]
    num_classes = predict_config["num_classes"]
    keep_aspect_ratio = predict_config.get("keep_aspect_ratio", True)
    patch_size = predict_config["patch_size"]
    model_path = predict_config["model_path"]

    # Define preprocessing parameters
    normalization_scheme = [norm_op] * len(modality_paths)
    target_spacing = [1.0, 1.0, 1.0]  # Isotropic 1mm spacing
    target_orientation = "RAS"

    # Apply preprocessing
    case_preprocessed, case_properties = preprocess_case_for_inference(
        crop_to_nonzero=crop_to_nonzero,
        images=images,
        intensities=None,  # Use default intensity normalization
        normalization_scheme=normalization_scheme,
        patch_size=patch_size,
        target_size=None,  # We use target_spacing instead
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        allow_missing_modalities=False,
        keep_aspect_ratio=keep_aspect_ratio,
        transpose_forward=[0, 1, 2],  # Standard transpose order
    )

    # Load the model checkpoint directly with Lightning
    if task_type == "segmentation":
        model = SupervisedSegModel.load_from_checkpoint(checkpoint_path=model_path)
    elif task_type == "classification":
        model = SupervisedClsModel.load_from_checkpoint(checkpoint_path=model_path)
    elif task_type == "regression":
        model = SupervisedRegModel.load_from_checkpoint(checkpoint_path=model_path)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Set model to evaluation mode
    model.eval()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    case_preprocessed = case_preprocessed.to(device)

    # Run inference
    with torch.no_grad():
        # Set up sliding window parameters
        overlap = 0.5  # Standard overlap for sliding window

        # Get prediction
        predictions = model.model.predict(
            data=case_preprocessed,
            mode="3D",
            mirror=False,  # No test-time augmentation
            overlap=overlap,
            patch_size=patch_size,
            sliding_window_prediction=True,
            device=device,
        )

    # Reverse preprocessing
    transpose_forward = [0, 1, 2]
    transpose_backward = [0, 1, 2]

    predictions_original, _ = reverse_preprocessing(
        crop_to_nonzero=crop_to_nonzero,
        images=predictions,
        image_properties=case_properties,
        n_classes=num_classes,
        transpose_forward=transpose_forward,
        transpose_backward=transpose_backward,
    )

    # Convert prediction to appropriate format based on task type
    if task_type == "classification":
        # For classification, apply softmax and take argmax
        predictions_softmax = torch.nn.functional.softmax(
            torch.from_numpy(predictions_original), dim=1
        )
        prediction_final = torch.argmax(predictions_softmax, dim=1)[0].numpy()
    elif task_type == "regression":
        # For regression, just take the raw prediction
        prediction_final = predictions_original[0, 0]
    else:  # Segmentation
        # For segmentation, apply argmax
        prediction_final = np.argmax(predictions_original[0], axis=0)

    # Save the prediction
    save_prediction(prediction_final, images[0], output_path)

    return output_path
