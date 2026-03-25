"""Model definition for OCT en face image quality assessment."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights


DEFAULT_DROPOUT = 0.5
DEFAULT_NUM_OUTPUTS = 1


def build_model(
    weights: EfficientNet_V2_L_Weights | None = None,
    dropout: float = DEFAULT_DROPOUT,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
) -> nn.Module:
    """
    Build the EfficientNetV2-L architecture used for OCT en face quality assessment.

    Parameters
    ----------
    weights:
        Optional torchvision EfficientNetV2-L pretrained weights.
        Use ``EfficientNet_V2_L_Weights.IMAGENET1K_V1`` for ImageNet initialization
        during fine-tuning, or ``None`` for inference-time reconstruction before loading
        the saved checkpoint.
    dropout:
        Dropout probability in the classifier head.
    num_outputs:
        Number of output neurons. For this binary task, this should remain 1.

    Returns
    -------
    nn.Module
        EfficientNetV2-L model with the notebook-matched classifier head.
    """
    model = models.efficientnet_v2_l(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_outputs),
    )
    return model


def build_training_model(dropout: float = DEFAULT_DROPOUT) -> nn.Module:
    """
    Convenience helper matching the original notebook fine-tuning setup.
    """
    return build_model(
        weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        dropout=dropout,
        num_outputs=1,
    )


def build_inference_model(dropout: float = DEFAULT_DROPOUT) -> nn.Module:
    """
    Convenience helper matching checkpoint-loading for inference/evaluation.
    """
    return build_model(weights=None, dropout=dropout, num_outputs=1)
