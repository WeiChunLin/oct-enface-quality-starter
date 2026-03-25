"""Evaluation helpers for OCT en face image quality assessment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .infer import DEFAULT_THRESHOLD, get_device, load_model
from .dataset import OCTEnfaceDataset
from .transforms import get_inference_transform


TARGET_NAMES = ["not acceptable", "acceptable"]


@torch.inference_mode()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on a DataLoader and return (y_true, y_prob).
    """
    all_logits = []
    all_targets = []

    for images, targets in tqdm(loader):
        images = images.to(device)
        logits = model(images).squeeze(1).detach().cpu().numpy()
        all_logits.extend(logits)
        all_targets.extend(targets.numpy())

    y_true = np.array(all_targets, dtype=int)
    y_prob = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    return y_true, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Compute binary classification metrics using the released threshold convention.
    """
    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=TARGET_NAMES,
            digits=4,
        ),
    }
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC = {auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | Path | None = None) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve (AUPRC = {auprc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    plt.figure()
    plt.imshow(conf_mat, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j], ha="center", va="center")
    plt.xticks([0, 1], TARGET_NAMES)
    plt.yticks([0, 1], TARGET_NAMES)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def evaluate_dataset(
    checkpoint_path: str | Path,
    image_paths: list[str | Path],
    labels: list[int],
    batch_size: int = 16,
    num_workers: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | None = None,
) -> dict:
    """
    End-to-end evaluation from image paths and labels.
    """
    device = get_device(device)
    model, device = load_model(checkpoint_path=checkpoint_path, device=device)

    dataset = OCTEnfaceDataset(
        image_list=image_paths,
        labels=labels,
        transform=get_inference_transform(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true, y_prob = predict_loader(model=model, loader=loader, device=device)
    metrics = compute_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    metrics["y_true"] = y_true
    metrics["y_prob"] = y_prob
    return metrics
