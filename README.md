# OCT en face image quality assessment

Starter repo files for an **inference/evaluation-only** release of a fine-tuned **EfficientNetV2-L** model for **OCT en face image quality assessment**.

## Released task

Binary classification:

- **1 = acceptable**
- **0 = not acceptable**

The model outputs a single logit. During inference:

- sigmoid is applied to obtain `probability_acceptable`
- the default decision threshold is **0.5**
- prediction is **acceptable** if `probability_acceptable > 0.5`

## Final released preprocessing

These settings match the released model assumptions:

- input image size: **480 × 480**
- images are loaded as **grayscale**
- grayscale is kept as **1 channel**
- normalization uses:
  - mean = `[0.5]`
  - std = `[0.5]`
- the normalized 1-channel tensor is repeated to **3 channels**
  so it can be passed to EfficientNetV2-L

The transform source is **Cell 12** from the original notebook.

## Repo starter files included

- `src/oct_quality/model.py`  
  EfficientNetV2-L architecture with the released binary classifier head

- `src/oct_quality/transforms.py`  
  Released inference transform and notebook-matched training transform

- `src/oct_quality/dataset.py`  
  Dataset helper for loading OCT en face image files

- `src/oct_quality/infer.py`  
  Checkpoint loading, single-image inference, and folder inference

- `src/oct_quality/evaluate.py`  
  Evaluation helpers, metrics, and plotting functions

- `config.yaml`  
  Centralized released model assumptions

## Expected checkpoint architecture

The released checkpoint should match:

- backbone: `torchvision.models.efficientnet_v2_l`
- classifier:
  - `Dropout(p=0.5)`
  - `Linear(1280, 1)`

Supported checkpoint formats:

- raw `state_dict`
- dictionary containing `state_dict`
- dictionary containing `model_state_dict`

If checkpoint keys include a `module.` prefix, it is removed automatically.

## Suggested repo structure

```text
oct-enface-quality/
├── README.md
├── config.yaml
└── src/
    └── oct_quality/
        ├── model.py
        ├── transforms.py
        ├── dataset.py
        ├── infer.py
        └── evaluate.py
```

## Minimal installation

Example package dependencies:

```bash
pip install torch torchvision pandas pillow matplotlib scikit-learn tqdm
```

If you want to load `config.yaml` directly in code, also install:

```bash
pip install pyyaml
```

## Example: predict one image

```python
from oct_quality.infer import predict_image

result = predict_image(
    checkpoint_path="checkpoints/model_best.pth",
    image_path="example.png",
)
print(result)
```

Example output:

```python
{
    "image_path": "example.png",
    "probability_acceptable": 0.8731,
    "pred_binary": 1,
    "pred_label": "acceptable",
    "threshold": 0.5,
}
```

## Example: predict a folder

```python
from oct_quality.infer import predict_folder

df = predict_folder(
    checkpoint_path="checkpoints/model_best.pth",
    image_dir="sample_images",
    output_csv="predictions.csv",
)
print(df.head())
```

Output columns:

- `image_path`
- `probability_acceptable`
- `pred_binary`
- `pred_label`

## Example: evaluate a labeled dataset

```python
from oct_quality.evaluate import evaluate_dataset

image_paths = [
    "img1.png",
    "img2.png",
    "img3.png",
]
labels = [1, 0, 1]

metrics = evaluate_dataset(
    checkpoint_path="checkpoints/model_best.pth",
    image_paths=image_paths,
    labels=labels,
    batch_size=8,
)

print("ROC AUC:", metrics["roc_auc"])
print("AUPRC:", metrics["auprc"])
print(metrics["classification_report"])
```

## Important notes before publishing

1. Keep the checkpoint and preprocessing exactly aligned.
2. State clearly that **positive = acceptable**.
3. Confirm all shared images are de-identified and permitted for release.
4. Add a license for both the code and the model weights.
5. Add a citation section once your manuscript/preprint is ready.

## Recommended next files to add later

For a fuller public repo, the next useful files would be:

- `requirements.txt`
- `pyproject.toml`
- `LICENSE`
- `__init__.py`
- example scripts or a CLI
- a `model_card.md`
