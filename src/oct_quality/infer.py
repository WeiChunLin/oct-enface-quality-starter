from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from .model import build_model
from .transforms import get_inference_transform

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
IMAGE_SIZE = 480
DEFAULT_THRESHOLD = 0.5
DEFAULT_CLASS_MAP = {
    0: "not acceptable",
    1: "acceptable",
}
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _resolve_device(device: str | None = None) -> str:
    """
    Resolve which device to use for inference.

    Parameters
    ----------
    device : str | None
        Requested device, such as "cpu" or "cuda".
        If None, CUDA is used when available; otherwise CPU.

    Returns
    -------
    str
        Resolved device string.
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_threshold(threshold: float) -> None:
    """
    Validate that threshold is between 0 and 1.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")


def _ensure_csv_parent(output_csv: str | Path) -> Path:
    """
    Ensure the parent folder for a CSV output path exists.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    return output_csv


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """
    Extract a model state_dict from a loaded checkpoint object.

    Supports:
    - raw state_dict
    - checkpoint dict with key 'state_dict'
    - checkpoint dict with key 'model_state_dict'

    Parameters
    ----------
    checkpoint : object
        Object returned by torch.load().

    Returns
    -------
    dict[str, torch.Tensor]
        Cleaned state_dict for model loading.

    Raises
    ------
    ValueError
        If the checkpoint format is unsupported.
    """
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected a raw state_dict, "
                "or a dict containing 'state_dict' or 'model_state_dict'."
            )
    else:
        raise ValueError(
            "Unsupported checkpoint object returned by torch.load(). "
            "Expected a dictionary-like checkpoint."
        )

    # Remove 'module.' prefix if model was saved with DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("module.", "")] = v

    return cleaned_state_dict


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Load a checkpoint file and return a clean state_dict.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a .pth checkpoint file.
    device : str | None
        Device mapping for torch.load().

    Returns
    -------
    dict[str, torch.Tensor]
        Clean state_dict ready for model.load_state_dict().

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    ValueError
        If the checkpoint format is unsupported.
    RuntimeError
        If torch.load() fails.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint path is not a file: {checkpoint_path}")

    resolved_device = _resolve_device(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}\n{e}") from e

    return _extract_state_dict(checkpoint)


def load_model(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[torch.nn.Module, str]:
    """
    Load the EfficientNetV2 model and checkpoint weights for inference.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained model checkpoint (.pth).
    device : str | None
        Device for inference. If None, CUDA is used when available,
        otherwise CPU.

    Returns
    -------
    tuple[torch.nn.Module, str]
        Loaded model in eval mode, and resolved device string.

    Raises
    ------
    RuntimeError
        If weights cannot be loaded into the model.
    """
    resolved_device = _resolve_device(device)

    model = build_model()
    state_dict = load_checkpoint_state_dict(checkpoint_path, device=resolved_device)

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            "Failed to load state_dict into model. "
            "Please confirm that the checkpoint matches the released architecture."
        ) from e

    model = model.to(resolved_device)
    model.eval()

    return model, resolved_device


def _predict_single_image_tensor(
    model: torch.nn.Module,
    image_path: str | Path,
    device: str,
    threshold: float = DEFAULT_THRESHOLD,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """
    Internal helper to predict a single image using a loaded model.
    """
    _validate_threshold(threshold)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported image extension: {image_path.suffix}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTS)}"
        )

    try:
        image = Image.open(image_path).convert("L")
    except UnidentifiedImageError as e:
        raise ValueError(f"Could not identify image file: {image_path}") from e
    except OSError as e:
        raise ValueError(f"Failed to open image file: {image_path}") from e

    transform = get_inference_transform(image_size=image_size)
    x = transform(image).unsqueeze(0).to(device)

    logit = model(x).squeeze(1)
    prob = torch.sigmoid(logit).item()
    pred = int(prob > threshold)

    return {
        "image_path": str(image_path),
        "probability_acceptable": float(prob),
        "pred_binary": pred,
        "pred_label": DEFAULT_CLASS_MAP[pred],
        "threshold": float(threshold),
    }


def _save_df_if_requested(
    df: pd.DataFrame,
    save_csv: bool = False,
    output_csv: str | Path | None = None,
) -> None:
    """
    Save a DataFrame to CSV if requested.

    Behavior
    --------
    - If save_csv is False: do nothing.
    - If save_csv is True and output_csv is None: save to 'predictions.csv'.
    - If save_csv is True and output_csv is provided: save to that path.
    """
    if not save_csv:
        return

    csv_path = Path("predictions.csv") if output_csv is None else _ensure_csv_parent(output_csv)
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------
# Public prediction functions
# ---------------------------------------------------------------------
@torch.inference_mode()
def predict_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    image_size: int = IMAGE_SIZE,
    device: str | None = None,
    save_csv: bool = False,
    output_csv: str | Path | None = None,
) -> dict:
    """
    Predict a single OCT en face image.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained model checkpoint (.pth).
    image_path : str | Path
        Path to a single image file.
    threshold : float, default=0.5
        Probability threshold for binary classification.
        Probability > threshold is labeled as acceptable (1).
    image_size : int, default=480
        Target input size for inference preprocessing.
    device : str | None, default=None
        Device for inference. If None, CUDA is used when available,
        otherwise CPU.
    save_csv : bool, default=False
        Whether to save the prediction result as a one-row CSV file.
    output_csv : str | Path | None, default=None
        CSV path to save the result. If save_csv=True and output_csv is None,
        the default filename 'predictions.csv' is used.

    Returns
    -------
    dict
        Prediction dictionary with:
        - image_path
        - probability_acceptable
        - pred_binary
        - pred_label
        - threshold
    """
    model, resolved_device = load_model(checkpoint_path=checkpoint_path, device=device)

    result = _predict_single_image_tensor(
        model=model,
        image_path=image_path,
        device=resolved_device,
        threshold=threshold,
        image_size=image_size,
    )

    if save_csv:
        df = pd.DataFrame([result])
        _save_df_if_requested(df, save_csv=True, output_csv=output_csv)

    return result


@torch.inference_mode()
def predict_paths(
    checkpoint_path: str | Path,
    image_paths: Iterable[str | Path],
    threshold: float = DEFAULT_THRESHOLD,
    image_size: int = IMAGE_SIZE,
    device: str | None = None,
    save_csv: bool = False,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Predict a batch of image paths and return a pandas DataFrame.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained model checkpoint (.pth).
    image_paths : Iterable[str | Path]
        Iterable of image paths to run inference on.
    threshold : float, default=0.5
        Probability threshold for binary classification.
        Probability > threshold is labeled as acceptable (1).
    image_size : int, default=480
        Target input size for inference preprocessing.
    device : str | None, default=None
        Device for inference. If None, CUDA is used when available,
        otherwise CPU.
    save_csv : bool, default=False
        Whether to save the prediction results as a CSV file.
    output_csv : str | Path | None, default=None
        CSV path to save the results. If save_csv=True and output_csv is None,
        the default filename 'predictions.csv' is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per image and columns:
        - image_path
        - probability_acceptable
        - pred_binary
        - pred_label
        - threshold

    Raises
    ------
    ValueError
        If image_paths is empty.
    """
    image_paths = list(image_paths)
    if len(image_paths) == 0:
        raise ValueError("image_paths is empty. Please provide at least one image path.")

    model, resolved_device = load_model(checkpoint_path=checkpoint_path, device=device)

    rows = []
    for image_path in image_paths:
        row = _predict_single_image_tensor(
            model=model,
            image_path=image_path,
            device=resolved_device,
            threshold=threshold,
            image_size=image_size,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    _save_df_if_requested(df, save_csv=save_csv, output_csv=output_csv)

    return df


def find_images(folder: str | Path) -> list[Path]:
    """
    Recursively find supported image files in a folder.

    Parameters
    ----------
    folder : str | Path
        Folder containing images.

    Returns
    -------
    list[Path]
        Sorted list of image file paths.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or is not a directory.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {folder}")

    image_paths = sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    )

    return image_paths


@torch.inference_mode()
def predict_folder(
    checkpoint_path: str | Path,
    image_dir: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    image_size: int = IMAGE_SIZE,
    device: str | None = None,
    save_csv: bool = False,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Predict all supported images in a folder recursively.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the trained model checkpoint (.pth).
    image_dir : str | Path
        Folder containing images.
    threshold : float, default=0.5
        Probability threshold for binary classification.
        Probability > threshold is labeled as acceptable (1).
    image_size : int, default=480
        Target input size for inference preprocessing.
    device : str | None, default=None
        Device for inference. If None, CUDA is used when available,
        otherwise CPU.
    save_csv : bool, default=False
        Whether to save the prediction results as a CSV file.
    output_csv : str | Path | None, default=None
        CSV path to save the results. If save_csv=True and output_csv is None,
        the default filename 'predictions.csv' is used.

    Returns
    -------
    pd.DataFrame
        Prediction results for all discovered images.

    Raises
    ------
    FileNotFoundError
        If the folder is missing or contains no supported images.
    """
    image_paths = find_images(image_dir)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No supported image files found in: {image_dir}")

    df = predict_paths(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        threshold=threshold,
        image_size=image_size,
        device=device,
        save_csv=save_csv,
        output_csv=output_csv,
    )
    return df