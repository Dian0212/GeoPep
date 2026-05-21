# Model Weights

Place trained ESM3-KAN checkpoints here. The per-residue model is large
(~16 GB), so this directory is gitignored except for this README.

## Expected File

| Model | Filename | Architecture |
|-------|----------|--------------|
| GeoPep per-residue | `model_distanceLoss.ckpt` | Per-residue head: 1536 → ... → 3 per position |

## Usage

Inference (one-command pipeline):

```bash
cd scripts
python inference_pipeline.py \
    --pdb-dir /path/to/pdb \
    --checkpoint ../model_weights/model_distanceLoss.ckpt
```

Inference (step-by-step, reading `prediction.checkpoint_path` from `configs/config.yaml`):

```bash
cd scripts
python predict_esm3.py --config ../configs/config.yaml
```
