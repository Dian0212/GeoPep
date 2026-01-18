# Model Weights

Place your trained ESM3-KAN checkpoint here.

## Expected File

| Model | Filename |
|-------|----------|
| ESM3-KAN | `model_distanceLoss.ckpt` |

## Usage

```bash
python scripts/predict_esm3.py \
    --checkpoint model_weights/model_distanceLoss.ckpt \
    --input data.json
```
