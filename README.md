# OCT image quality assessment (Zeiss Cirrus ONH)

This repository provides an **inference/evaluation-only** release of a
fine-tuned **EfficientNetV2-L** model for **OCT en face image quality
assessment**.

## Task

This model performs binary classification of OCT en face image quality:

-   **1 = acceptable**
-   **0 = not acceptable**

The model outputs a single logit. During inference:

-   a sigmoid is applied to obtain `probability_acceptable`
-   the default decision threshold is **0.5**
-   prediction is **acceptable** if `probability_acceptable > 0.5`

## Preprocessing

The released model uses the following preprocessing:

-   input image size: **480 × 480**
-   images are loaded as **grayscale**
-   normalization:
    -   mean = `[0.5]`
    -   std = `[0.5]`
-   the normalized 1-channel tensor is repeated to **3 channels** before
    being passed to EfficientNetV2-L

## Repository contents

-   `src/oct_quality/model.py`
-   `src/oct_quality/transforms.py`
-   `src/oct_quality/dataset.py`
-   `src/oct_quality/infer.py`
-   `src/oct_quality/evaluate.py`
-   `config.yaml`

## Model architecture

-   backbone: `torchvision.models.efficientnet_v2_l`
-   classifier:
    -   `Dropout(p=0.5)`
    -   `Linear(1280, 1)`

## Installation

``` bash
pip install -r requirements.txt
pip install -e .
```

## Checkpoint

The released checkpoint file is:

    checkpoints/EffiNetV2_OCT_quality_Assess_2026.pth

This repository uses Git LFS for the checkpoint file.\
After cloning, make sure Git LFS is installed so the model weights are
downloaded correctly.

## Example: predict one image

``` python
from oct_quality.infer import predict_image

result = predict_image(
    checkpoint_path="checkpoints/EffiNetV2_OCT_quality_Assess_2026.pth",
    image_path="test_images/sample.png",
)

print(result)
```

### Example output

``` python
{
    "image_path": "test_images/sample.png",
    "probability_acceptable": 0.8731,
    "pred_binary": 1,
    "pred_label": "acceptable",
    "threshold": 0.5,
}
```

## Example: predict one image and save CSV

``` python
from oct_quality.infer import predict_image

result = predict_image(
    checkpoint_path="checkpoints/EffiNetV2_OCT_quality_Assess_2026.pth",
    image_path="test_images/sample.png",
    save_csv=True,
    output_csv="results/single_prediction.csv",
)
```

## Example: predict a folder

``` python
from oct_quality.infer import predict_folder

df = predict_folder(
    checkpoint_path="checkpoints/EffiNetV2_OCT_quality_Assess_2026.pth",
    image_dir="test_images",
    save_csv=True,
    output_csv="results/folder_predictions.csv",
)

print(df.head())
```

## Output columns

-   `image_path`
-   `probability_acceptable`
-   `pred_binary`
-   `pred_label`
-   `threshold`

## Notes

-   This release is intended for **inference and evaluation only**.
-   Training and fine-tuning code are not included in this repository.
-   Input images should match the expected OCT en face format used for
    model development.


