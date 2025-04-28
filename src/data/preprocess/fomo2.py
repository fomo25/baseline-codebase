import os
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
    save_pickle,
)
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from data.task_configs import task2_config
from utils.utils import parallel_process


def process_subject(task_info):
    """
    Process a single subject for Task 2.

    Args:
        task_info: A tuple containing (subject, source_path, images_dir, labels_dir, pp_config, target_preprocessed, prefix)

    Returns:
        Success message or error message
    """
    (
        subject,
        source_path,
        images_dir,
        labels_dir,
        pp_config,
        target_preprocessed,
        prefix,
    ) = task_info
    modalities = pp_config["modalities"]

    try:
        session_path = join(images_dir, subject, "ses_1")
        label_path = join(labels_dir, subject, "ses_1", "seg.nii.gz")

        if not os.path.exists(session_path) or not os.path.exists(label_path):
            return f"Error: Missing data for {subject}"

        # Get all modality images
        image_files = []
        modality_mapping = {}

        for file in os.listdir(session_path):
            if not file.endswith(".nii.gz"):
                continue

            # Determine modality
            if "dwi" in file.lower():
                modality_index = 0  # DWI
            elif "flair" in file.lower():
                modality_index = 1  # T2FLAIR
            elif "swi" in file.lower() or "t2s" in file.lower():
                modality_index = 2  # SWI_OR_T2STAR
            else:
                return f"Warning: Skipping file {file}"

            source_img = join(session_path, file)
            image_files.append(source_img)
            modality_mapping[modality_index] = source_img

        # Skip if we don't have all required modalities
        if len(image_files) < len(modalities):
            return f"Error: Not all modalities found for {subject}"

        # Load images for preprocessing
        images = [
            nib.load(modality_mapping[i])
            for i in range(len(modalities))
            if i in modality_mapping
        ]

        # Load segmentation label
        label = nib.load(label_path)

        # Apply preprocessing with label
        preprocessed_data, preprocessed_label, properties = (
            preprocess_case_for_training_with_label(
                images=images,
                label=label,
                normalization_operation=[
                    pp_config["norm_op"] for _ in pp_config["modalities"]
                ],
                allow_missing_modalities=False,
                crop_to_nonzero=pp_config["crop_to_nonzero"],
                keep_aspect_ratio_when_using_target_size=pp_config["keep_aspect_ratio"],
            )
        )

        data_with_label = preprocessed_data + [preprocessed_label]

        # Save preprocessed data
        save_path = join(target_preprocessed, f"{prefix}_{subject}")
        np.save(save_path + ".npy", np.array(data_with_label, dtype=object))

        # Save properties as pickle
        save_pickle(properties, save_path + ".pkl")

        return f"Processed {subject}"

    except Exception as e:
        return f"Error processing {subject}: {str(e)}"


def convert_and_preprocess_task2(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess all subjects for Task 2 in parallel.

    Args:
        source_path: Path to the source data directory
        output_path: Path where preprocessed data will be saved (optional)
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    # Get configuration from task2_config
    pp_config = task2_config
    task_name = pp_config["task_name"]
    prefix = "FOMO2"

    # Input data paths
    labels_dir = join(source_path, "labels")
    images_dir = join(source_path, "preprocessed")

    # Output path for preprocessed data
    target_preprocessed = join(output_path, task_name)

    # Create directory
    ensure_dir_exists(target_preprocessed)

    # Collect subjects to process
    subjects = sorted(os.listdir(images_dir))

    # Create task information for each subject
    tasks = [
        (
            subject,
            source_path,
            images_dir,
            labels_dir,
            pp_config,
            target_preprocessed,
            prefix,
        )
        for subject in subjects
    ]

    # Process all subjects in parallel
    parallel_process(
        process_subject, tasks, num_workers, desc="Processing subjects for Task 2"
    )

    print(f"Task 2 preprocessing completed. Data saved to {target_preprocessed}")
