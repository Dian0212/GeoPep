# GeoPep

**Geometric-aware Peptide-Protein Binding Site Prediction**

GeoPep predicts which residues of a protein bind a given peptide. It combines the **ESM3** protein foundation model with **Kolmogorov-Arnold Networks (KANs)** in a **per-residue** classification head, trained with a differentiable distance loss that injects 3D geometric information.

---

## Table of Contents

1. [Quick Start (one command)](#quick-start)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [HuggingFace Token](#huggingface-token)
5. [Configuration](#configuration)
6. [End-to-End Pipelines](#end-to-end-pipelines)
7. [Step-by-Step Usage](#step-by-step-usage)
8. [Data Format](#data-format)
9. [Architecture](#architecture)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

After [installation](#installation) and [HF token setup](#huggingface-token):

```bash
cd scripts

# Train a model from a folder of PDBs (folder must contain complex/ and interface/ subdirs)
python train_pipeline.py --pdb-dir /path/to/pdb

# Run inference on a folder of PDBs using a trained checkpoint
python inference_pipeline.py \
    --pdb-dir /path/to/pdb \
    --checkpoint ../model_weights/model_distanceLoss.ckpt
```

The training pipeline saves a checkpoint to `model_weights/`. The inference pipeline writes per-residue binding probabilities to `result/predictions.json`.

---

## Installation

```bash
git clone https://github.com/Dian0212/GeoPep.git
cd GeoPep
conda create -n geopep python=3.10
conda activate geopep
pip install -r requirements.txt
```

Copy the config template:

```bash
cp configs/config.yaml.template configs/config.yaml
# Edit configs/config.yaml — only the huggingface.token field is strictly required
```

---

## Project Structure

```
GeoPep/
├── geopep/                       # Library code
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # PeptideComplexDataset (used by train.py)
│   ├── models/
│   │   ├── __init__.py
│   │   └── esm3_kan.py           # ESM3-KAN per-residue model
│   └── hf_auth.py                # HuggingFace token resolver (env > config > prompt)
│
├── scripts/                      # Runnable entry points
│   ├── preprocess.py             # PDB -> JSON
│   ├── train.py                  # Single-step trainer
│   ├── predict_esm3.py           # Single-step inference
│   ├── postprocess.py            # Softmax -> binary + result JSON / CSV
│   ├── train_pipeline.py         # ONE COMMAND: PDB folder -> trained model
│   └── inference_pipeline.py     # ONE COMMAND: PDB folder + ckpt -> predictions.json
│
├── configs/
│   ├── config.yaml.template      # Tracked: copy this to config.yaml
│   └── config.yaml               # GITIGNORED: your local copy with your token
│
├── model_weights/
│   ├── README.md
│   └── model_distanceLoss.ckpt   # Drop your trained checkpoint here (~16 GB)
│
├── pdb/                          # Input PDBs
│   ├── complex/
│   └── interface/
├── json/                         # Preprocessed JSON outputs (gitignored)
├── result/                       # Inference results (gitignored)
├── requirements.txt
└── README.md
```

---

## HuggingFace Token

**ESM3 is a gated model.** You need a HuggingFace token with access to the `EvolutionaryScale/esm3` repository.

1. Create a token at https://huggingface.co/settings/tokens (Read access is enough)
2. On the [ESM3 model page](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) click "Agree and access repository" while logged in
3. Provide the token to GeoPep via **one of these methods** (the scripts try them in order):

   **a) Environment variable (recommended for CI / shared machines):**
   ```bash
   export HF_TOKEN="hf_xxx..."          # Linux/macOS
   $env:HF_TOKEN="hf_xxx..."            # PowerShell
   ```

   **b) Config file (local development):**
   ```yaml
   # configs/config.yaml
   huggingface:
     token: "hf_xxx..."
   ```
   `configs/config.yaml` is **gitignored** — don't worry about leaking it. But never paste a real token into `config.yaml.template` (which IS tracked).

   **c) Interactive prompt:** if neither of the above is set, the script will pause and ask for your token on stdin.

> ⚠️ **If a token has ever appeared in a public git commit, treat it as compromised** — revoke it at https://huggingface.co/settings/tokens and generate a new one.

---

## Configuration

All settings live in `configs/config.yaml` (copy from `config.yaml.template`). Both the one-command pipelines and the individual scripts read from the same file. CLI flags override the relevant fields at runtime.

Key sections:

| Section | Purpose |
|---|---|
| `preprocess.*`     | Paths for PDB → JSON conversion |
| `data.*`           | train/val JSON shards |
| `model.*`          | peptide_len (50), protein_len (500), num_label_types (3) |
| `training.*`       | batch_size, learning_rate, max_epochs, distance loss toggle, `checkpoint_dir` |
| `prediction.*`     | checkpoint_path, input_json, device (cuda/cpu) |
| `hardware.*`       | GPU ids, mixed-precision setting |
| `huggingface.token`| HF token (or leave placeholder and use env var / prompt) |

---

## End-to-End Pipelines

### Train pipeline

```bash
cd scripts
python train_pipeline.py --pdb-dir /path/to/pdb
```

This runs preprocessing → training → checkpoint save. Required layout:

```
/path/to/pdb/
├── complex/                # Full peptide-protein complexes
│   ├── 1abc_A_B.pdb        # Naming: PDBID_PeptideChain_ProteinChain.pdb
│   └── ...
└── interface/              # Interface-only PDBs (same filenames)
    ├── 1abc_A_B.pdb
    └── ...
```

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--pdb-dir`         | required             | PDB root folder |
| `--output-dir`      | `../model_weights`   | Where to save the trained `.ckpt` |
| `--work-dir`        | `../json/preprocessed` | Intermediate preprocessed JSONs |
| `--config`          | `../configs/config.yaml` | Base config (hyperparameters, HF token) |
| `--val-ratio`       | `0.2`                | Fraction of JSON shards used as val |
| `--skip-preprocess` | off                  | Reuse existing preprocessed JSONs |

### Inference pipeline

```bash
cd scripts
python inference_pipeline.py \
    --pdb-dir /path/to/pdb \
    --checkpoint ../model_weights/model_distanceLoss.ckpt
```

Runs preprocessing (inference-only mode — no interface labels needed) → prediction → result JSON. The PDB folder can be flat OR have a `complex/` subdir.

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--pdb-dir`         | required             | PDB folder |
| `--checkpoint`      | required             | Trained model `.ckpt` |
| `--result-dir`      | `../result`          | Where to write `predictions.json` |
| `--work-dir`        | `../json/inference`  | Intermediate preprocessed JSON |
| `--config`          | `../configs/config.yaml` | Base config |
| `--device`          | from config (cuda)   | `cuda` or `cpu` |
| `--skip-preprocess` | off                  | Reuse intermediate JSON |

The output is a single `result/predictions.json` keyed by `{pdb_id}_{chain_key}`:

```json
{
  "1a1r_C_A": {
    "peptide_chain": "GSVVIVGRIVLSGKPA",
    "protein_chain": "VEGEVQIVSTATQTFLAT...",
    "peptide_bindingProbability": "0.99 0.99 0.99 ...",
    "protein_bindingProbability": "0.99 0.19 0.01 ..."
  }
}
```

Probability counts match residue counts exactly (padding is stripped). Each value is the **raw class-1 (interface) probability** from the model's softmax.

---

## Step-by-Step Usage

If you prefer running the individual stages (debugging, custom pipelines, etc.):

```bash
cd scripts

# 1. Preprocess PDB files to JSON (per-shard splits)
python preprocess.py --config ../configs/config.yaml

# 2. Train
python train.py --config ../configs/config.yaml

# 3. Predict (writes model_out_argmax / model_out_softmax back into the input JSON)
python predict_esm3.py --config ../configs/config.yaml

# 4. Postprocess (optional: CSV per-residue + final result JSON)
python postprocess.py --input ../json/preprocessed/data_part_5.json --result-dir ../result
# Or: --output-dir ../csv_output --threshold 0.5  for per-PDB CSVs
```

---

## Data Format

### Input sequence layout

```
PEPTIDESEQ<pad><pad>...|PROTEINSEQ<pad><pad>...
|<------- 50 ------->|<-------- 500 ----------->|
```

| Position | Content |
|---|---|
| 0..49     | Peptide (left-padded with `<pad>` to 50) |
| 50        | Separator `|` |
| 51..550   | Protein (left-padded with `<pad>` to 500) |

### Label encoding (per residue)

| Value | Meaning |
|---|---|
| 0 | Non-interface residue |
| 1 | Interface (binding site) residue |
| 2 | Padding |
| 3 | Separator |

### Distance map

| Value | Meaning |
|---|---|
| `0`   | Interface residue itself |
| `>0`  | Normalized distance (0–10) to the nearest interface residue |
| `-1`  | Separator |
| `-2`  | Padding |

---

## Architecture

```
input tokens [B, 553]                ← BOS + 551 residue tokens + EOS
       │
   ESM3 encoder
       ↓ embeddings[:, 1:552, :]      drop BOS / EOS
   [B, 551, 1536]                     per-residue 1536-d embeddings
       ↓ reshape
   [B * 551, 1536]                    each residue is an independent sample
       ↓
   KAN_model1: 1536 → 1153
   KAN_model2: 1153 → 770
   KAN_model3: 770  → 387
   KAN_model4: 387  → 3
   KAN_model5: 3    → 3
       ↓ reshape + permute
   logits [B, 3, 551]                 3-class logits per residue
       ↓ softmax(dim=1)
   binding probability = softmax[:, 1, :]
```

### Loss = Cross-Entropy + Differentiable Distance Loss

For each batch, the loss has two terms:

1. **Weighted CE**: per-half cross-entropy with class weights `[0.2, 0.8, 0.0]` (padding ignored).
2. **Distance loss** (when `use_distance_loss: true`):
   `L_dist = Σ P_binding(i) · dist(i)  /  num_valid_residues`
   The model is penalized for predicting "binding" at residues far from the true interface.

Negative distances (padding/separator) are clamped to 0 before this term, and class-2 weight is 0 in CE — so padding positions never contribute gradient.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 Unauthorized` when downloading ESM3 | Token missing/invalid; re-check the HF token and that you accepted the model's TOS |
| `RuntimeError: CUDA out of memory` | Reduce `training.batch_size` or use `precision: 16` |
| `UnicodeEncodeError: 'gbk' codec ...` on Windows | Already mitigated in scripts; if it recurs, set `$env:PYTHONIOENCODING="utf-8"` |
| `IndexError: list index out of range` in train | Preprocessing produced 0 valid entries — check length filters (peptide ∈ [10, 50], protein ∈ [10, 500]) |
| Strict `load_state_dict` fails on inference | The checkpoint's KAN layer sizes don't match the inference module; the script falls back to `strict=False` and prints missing/unexpected keys |

---

## License

MIT
