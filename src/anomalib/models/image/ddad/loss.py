"""Loss functions for the DDAD model implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch import nn


class DDADLoss(nn.Module):
    """Overall loss function of the second training phase of the DSR model.

    The total loss consists of:
        - MSE loss between non-anomalous quantized input image and anomalous subspace-reconstructed
          non-quantized input (hi and lo)
        - MSE loss between input image and reconstructed image through object-specific decoder,
        - Focal loss between computed segmentation mask and ground truth mask.
    """

    def __init__(self, beta_start, beta_end, trajectory_steps) -> None:
        super().__init__()
        betas = np.linspace(
            beta_start,
            beta_end,
            trajectory_steps,
            dtype=np.float64,
        )
        b = torch.tensor(betas).type(torch.float)  # .to(config.model.device)

    def forward(
        self,
        output,
        e,
    ):
        """Compute the loss over a batch for the DSR model.

        Args:
            model (Tensor): Reconstructed non-quantized hi feature
            batch (Tensor): Reconstructed non-quantized lo feature
            t (Tensor): Non-defective quantized hi feature

        Returns:
            Tensor: Total loss
        """
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DomainAdaptationLoss(nn.Module):
    def __init__(self, config):
        super(CustomLossModule, self).__init__()
        self.config = config
        # Initialize CosineSimilarity directly as a class attribute
        self.cos_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, a, b, c, d):
        loss1 = 0
        loss2 = 0
        loss3 = 0
        for item in range(len(a)):
            # Use self.cos_loss directly
            loss1 += torch.mean(
                1 - self.cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)),
            )
            loss2 += (
                torch.mean(1 - self.cos_loss(b[item].view(b[item].shape[0], -1), c[item].view(c[item].shape[0], -1)))
                * self.config["model"]["DLlambda"]
            )
            loss3 += (
                torch.mean(1 - self.cos_loss(a[item].view(a[item].shape[0], -1), d[item].view(d[item].shape[0], -1)))
                * self.config["model"]["DLlambda"]
            )
        total_loss = loss1 + loss2 + loss3
        return total_loss
