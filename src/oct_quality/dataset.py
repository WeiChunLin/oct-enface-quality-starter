"""Dataset utilities for OCT en face image quality assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from torch.utils.data import Dataset


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class OCTEnfaceDataset(Dataset):
    """
    Dataset for loading OCT en face images.

    Parameters
    ----------
    image_list:
        Iterable of image filenames or paths.
    root_dir:
        Optional root directory prepended to relative image filenames.
    labels:
        Optional labels aligned to ``image_list``. Use for evaluation.
    transform:
        Optional image transform.
    strict_exists:
        If True, raise an error when an image path is missing. If False, silently
        skip missing files and keep only valid images.
    """
    def __init__(
        self,
        image_list: Iterable[str | Path],
        root_dir: str | Path | None = None,
        labels: Sequence[int] | None = None,
        transform=None,
        strict_exists: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform
        self.strict_exists = strict_exists

        image_list = list(image_list)
        if labels is not None and len(labels) != len(image_list):
            raise ValueError("labels must have the same length as image_list")

        self.samples: list[tuple[Path, int | None]] = []
        missing: list[Path] = []

        for idx, item in enumerate(image_list):
            path = Path(item)
            if self.root_dir is not None and not path.is_absolute():
                path = self.root_dir / path

            if path.suffix.lower() not in VALID_SUFFIXES:
                continue

            if not path.exists():
                missing.append(path)
                if strict_exists:
                    raise FileNotFoundError(f"Missing image file: {path}")
                continue

            label = None if labels is None else int(labels[idx])
            self.samples.append((path, label))

        if missing and not strict_exists:
            print(f"[Warning] Skipped {len(missing)} missing image files (e.g. {missing[:3]})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("L")  # true grayscale

        if self.transform is not None:
            image = self.transform(image)

        if label is None:
            return image, str(image_path)
        return image, label

    @property
    def image_paths(self) -> list[str]:
        return [str(path) for path, _ in self.samples]
