task1_config = {
    "task_name": "Task001_FOMO1",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("DWI", "T2FLAIR", "SWI_OR_T2STAR"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 2,
    "keep_aspect_ratio": True,
    "task_type": "classification",
    "label_extension": ".txt",
    "labels": {0: "Negative", 1: "Positive"},
}

task2_config = {
    "task_name": "Task002_FOMO2",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("DWI", "T2FLAIR", "SWI_OR_T2STAR"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 2,
    "keep_aspect_ratio": True,
    "task_type": "segmentation",
    "label_extension": ".txt",
    "labels": {0: "background", 1: "menigioma"},
}

task3_config = {
    "task_name": "Task003_FOMO3",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1", "T2"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 1,  # For regression, output dimension is 1
    "keep_aspect_ratio": True,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},  # Define as regression task
}
