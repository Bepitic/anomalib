"""Lightning Implementatation of the DDAD Model.

DDAD: Anomaly Detection with Conditioned Denoising Diffusion Models

Paper https://arxiv.org/abs/2305.15956
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms import transforms

from anomalib import LearningType
from anomalib.models.components import AnomalyModule
from anomalib.models.image.ddad.feature_extractor import DomainAdaptationModel
from anomalib.models.image.ddad.loss import DDADLoss, DomainAdaptationLoss
from anomalib.models.image.ddad.torch_model import UNetModel

logger = logging.getLogger(__name__)

__all__ = ["DDAD"]


class DDAD(AnomalyModule):
    """DDAD: Anomaly Detection with Conditioned Denoising Diffusion Models.

    Args:
        name (int): description.
            Defaults to ``1``.
    """

    def __init__(
        self,
        img_size,
        base_channels,
        conv_resample=True,
        n_heads=1,
        n_head_channels=-1,
        channel_mults='',
        num_res_blocks=2,
        dropout=0,
        attention_resolutions='32,16,8',
        biggan_updown=True,
        in_channels=1,
    ) -> None:
        super().__init__()
        self.model: UNetModel = UNetModel(
            img_size=img_size,
            base_channels=base_channels,
            conv_resample=conv_resample,
            n_heads=n_heads,
            n_head_channels=n_head_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            attention_resolutions=attention_resolutions,
            biggan_updown=biggan_updown,
            in_channels=in_channels,
        )
        trajectory_steps = 1000

        self.trajectory_steps = trajectory_steps

        self.loss = DDADLoss(beta_start=0.0001, beta_end=0.02, trajectory_steps=self.trajectory_steps)

        self.domain_adaptation_model = DomainAdaptationModel()
        self.loss_domain_adaptation = DomainAdaptationLoss()

    def on_train_start(self) -> None:
        """Initialize the centroid for the memory bank computation."""
        # self.model.initialize_centroid(data_loader=self.trainer.datamodule.train_dataloader()) this is from otehr model? TODO
        pass

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step for the DDAD model.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch input.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            STEP_OUTPUT: Loss value.
        """
        del args, kwargs  # These variables are not used.

        if self.train_unet:
            # take the batch size batch[0] the images shape[0] the batch size .long() no idea
            t = torch.randint(0, self.trajectory_steps, (batch[0].shape[0],)).long()

            x_0 = batch[0]

            e = torch.randn_like(x_0)
            at = (1 - self.b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

            x = at.sqrt() * x_0 + (1 - at).sqrt() * e
            output = self.model(x, t.float())

            loss = self.loss(output=output, e=e)

            return {"loss-unet": loss}
        else:
            half_batch_size = batch[0].shape[0] // 2
            target = batch[0][:half_batch_size]  # .to(config.model.device)
            input_x = batch[0][half_batch_size:]  # .to(config.model.device)

            reconst_fe, target_fe, target_frozen_fe, reconst_frozen_fe = self.domain_adaptation_model(
                target=target,
                input=input_x,
            )

            loss = self.loss_domain_adaptation(a=reconst_fe, b=target_fe, c=target_frozen_fe, d=reconst_frozen_fe)
            return {"loss-domain_adaptation": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step for the DDAD model.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict: Anomaly map computed by the model.
        """
        del args, kwargs  # These variables are not used.

        return batch

    def on_test_start(self) -> None:
        """Perform the start of test of DDAD model."""

        self.domain_adaptation_model.eval()

    def test_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        """Perform the test step of DDAD model."""

        labels_list = []
        predictions = []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        forward_list = []

        with torch.no_grad():
            for input, gt, labels in self.testloader:
                input = input.to(self.config.model.device)
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)

                anomaly_map = self.transform(anomaly_map)
                gt = self.transform(gt)

                forward_list.append(input)
                anomaly_map_list.append(anomaly_map)

                gt_list.append(gt)
                reconstructed_list.append(x0)
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == "good" else 1)
                    predictions.append(torch.max(pred).item())

        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)
        metric.optimal_threshold()
        if self.config.metrics.auroc:
            print("AUROC: ({:.1f},{:.1f})".format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
        if self.config.metrics.pro:
            print("PRO: {:.1f}".format(metric.pixel_pro() * 100))
        if self.config.metrics.misclassifications:
            metric.miscalssified()
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if not os.path.exists("results"):
            os.mkdir("results")
        if self.config.metrics.visualisation:
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        """Perform backward-pass for the DDAD model.

        Args:
            loss (torch.Tensor): Loss value.
            *args: Arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs  # These variables are not used.

        loss.backward()

    def on_epoch_end(self) -> None:
        """DDAD functionality when finishing epoch."""
        if self.current_epoch == self.half:
            self.domain_adaptation_model.setup_reconstruction(self.model)#, self.config)
            self.train_unet = False

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """CFA specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for the DDAD Model.

        Returns:
            Optimizer: Adam optimizer
        """
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            amsgrad=True,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
