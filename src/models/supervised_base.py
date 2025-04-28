from typing import Optional
import torch
from torch.optim import AdamW
import copy
import logging

import lightning as L
from yucca.pipeline.preprocessing import YuccaPreprocessor
from yucca.functional.utils.kwargs import filter_kwargs
from batchgenerators.utilities.file_and_folder_operations import join
from models import networks


class BaseSupervisedModel(L.LightningModule):
    """
    Base class for supervised models (segmentation, classification, regression).
    Implements common functionality and defines abstract methods that subclasses must implement.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        deep_supervision: bool = False,
    ):
        super().__init__()

        self.num_classes = config["num_classes"]
        self.num_modalities = config["num_modalities"]
        self.patch_size = config["patch_size"]
        self.plans = config.get("plans", {})
        self.model_name = config["model_name"]
        self.version_dir = config["version_dir"]
        self.task_type = config["task_type"]  # Added task_type property

        self.sliding_window_prediction = True
        self.sliding_window_overlap = 0.5  # nnUNet default
        self.test_time_augmentation = False
        self.progress_bar = True

        self.do_compile = do_compile
        self.compile_mode = compile_mode

        # Loss
        self.deep_supervision = deep_supervision

        # Optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.eps = eps
        self.betas = betas

        # Set up metrics in subclasses
        self.train_metrics = self._configure_metrics(prefix="train")
        self.val_metrics = self._configure_metrics(prefix="val")

        self.save_hyperparameters()
        self.load_model()

        self.model = (
            torch.compile(self.model, mode=self.compile_mode)
            if self.do_compile
            else self.model
        )

    def _configure_metrics(self, prefix: str):
        """
        Configure metrics specific to the task type.
        Must be implemented by subclasses.

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of metrics for the task
        """
        raise NotImplementedError("Subclasses must implement _configure_metrics")

    def _configure_losses(self):
        """
        Configure loss functions specific to the task type.
        Must be implemented by subclasses.

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        raise NotImplementedError("Subclasses must implement _configure_losses")

    def load_model(self):
        """Load the appropriate model architecture"""
        print(f"Loading Model: 3D {self.model_name}")
        model_class = getattr(networks, self.model_name)

        print("Found model class: ", model_class)

        conv_op = torch.nn.Conv3d
        norm_op = torch.nn.InstanceNorm3d
        print("MODALITIES", self.num_modalities)

        # Pass task_type directly to UNet without mapping
        model_kwargs = {
            # Applies to all models
            "input_channels": self.num_modalities,
            "num_classes": self.num_classes,
            "output_channels": self.num_classes,
            "deep_supervision": self.deep_supervision,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # MedNeXt
            "checkpoint_style": None,
            # ensure not pretraining
            "mode": self.task_type,  # Pass task_type directly
        }
        model_kwargs = filter_kwargs(model_class, model_kwargs)
        self.model = model_class(**model_kwargs)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Set up task-specific loss functions
        self.loss_fn_train, self.loss_fn_val = self._configure_losses()

        self.optim = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            eps=self.eps,
            betas=self.betas,
        )

        # Scheduler with early cut-off factor of 1.15
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=int(self.trainer.max_epochs * 1.15), eta_min=1e-9
        )

        # Return the optimizer and scheduler - the loss is not returned
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}

    def forward(self, inputs):
        """Forward pass through the model"""
        return self.model(inputs)

    def _process_batch(self, batch):
        """Process batch data - can be overridden by subclasses if needed"""
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        return inputs, target, file_path

    def training_step(self, batch, _batch_idx):
        """Training step"""
        inputs, target, _ = self._process_batch(batch)

        output = self(inputs)
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision and hasattr(output, "__iter__"):
            # If deep_supervision is enabled, output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output = output[0]
            target = target[0]

        metrics = self.compute_metrics(self.train_metrics, output, target)
        self.log_dict(
            {"train/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, _batch_idx):
        """Validation step"""
        inputs, target, _ = self._process_batch(batch)

        output = self(inputs)
        loss = self.loss_fn_val(output, target)
        metrics = self.compute_metrics(self.val_metrics, output, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def on_predict_start(self):
        """Set up for prediction"""
        self.preprocessor = YuccaPreprocessor(join(self.version_dir, "hparams.yaml"))

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        """Prediction step"""
        case, case_id = batch
        (
            case_preprocessed,
            case_properties,
        ) = self.preprocessor.preprocess_case_for_inference(
            case, self.patch_size, self.sliding_window_prediction
        )

        predictions = self.model.predict(
            data=case_preprocessed,
            mode="3D",
            mirror=self.test_time_augmentation,
            overlap=self.sliding_window_overlap,
            patch_size=self.patch_size,
            sliding_window_prediction=self.sliding_window_prediction,
            device=self.device,
        )
        predictions, case_properties = self.preprocessor.reverse_preprocessing(
            predictions, case_properties
        )
        return {
            "predictions": predictions,
            "properties": case_properties,
            "case_id": case_id[0],
        }

    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute task-specific metrics.
        Should be implemented/extended by subclasses for task-specific metrics.
        """
        raise NotImplementedError("Subclasses must implement compute_metrics")

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict with handling for different model architectures"""
        # First we filter out layers that have changed in size
        # This is often the case in the output layer.
        # If we are finetuning on a task with a different number of classes
        # than the pretraining task, the # output channels will have changed.
        old_params = copy.deepcopy(self.state_dict())
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if (k in old_params) and (old_params[k].shape == state_dict[k].shape)
        }
        rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        rejected_keys_shape = [
            k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape
        ]
        rejected_keys_data = []

        # Here there's also potential to implement custom loading functions.
        # E.g. to load 2D pretrained models into 3D by repeating or something like that.

        # Now keep track of the # of layers with succesful weight transfers
        successful = 0
        unsuccessful = 0
        super().load_state_dict(state_dict, *args, **kwargs)
        new_params = self.state_dict()
        for param_name, p1, p2 in zip(
            old_params.keys(), old_params.values(), new_params.values()
        ):
            # If more than one param in layer is NE (not equal) to the original weights we've successfully loaded new weights.
            if p1.data.ne(p2.data).sum() > 0:
                successful += 1
            else:
                unsuccessful += 1
                if (
                    param_name not in rejected_keys_new
                    and param_name not in rejected_keys_shape
                ):
                    rejected_keys_data.append(param_name)

        logging.warn(
            f"Succesfully transferred weights for {successful}/{successful+unsuccessful} layers"
        )
        logging.warn(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post check not succesful: {rejected_keys_data}."
        )

        return successful

    @staticmethod
    def create(task_type, config, **kwargs):
        """
        Factory method to create the appropriate model based on task type

        Args:
            task_type: Type of task (segmentation, classification, regression)
            config: Configuration dictionary
            **kwargs: Additional arguments for the model

        Returns:
            BaseSupervisedModel: Instance of appropriate model subclass
        """
        if task_type == "segmentation":
            from models.supervised_seg import SupervisedSegModel

            return SupervisedSegModel(config=config, **kwargs)
        elif task_type == "classification":
            from models.supervised_cls import SupervisedClsModel

            return SupervisedClsModel(config=config, **kwargs)
        elif task_type == "regression":
            from models.supervised_reg import SupervisedRegModel

            return SupervisedRegModel(config=config, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
