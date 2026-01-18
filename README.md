# GeoPep

**Geometric-aware Peptide-Protein Binding Site Prediction**

GeoPep predicts which residues in a protein will bind to a peptide. It combines ESM3 protein foundation model with Kolmogorov-Arnold Networks (KANs).

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
4. [Complete Pipeline](#complete-pipeline)
   - [Step 1: Preprocessing](#step-1-preprocessing-pdb--json)
   - [Step 2: Training](#step-2-training)
   - [Step 3: Prediction](#step-3-prediction)
   - [Step 4: Postprocessing](#step-4-postprocessing)
5. [Data Format](#data-format)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/GeoPep.git
cd GeoPep
```

### 2. Create conda environment (recommended)

```bash
conda create -n geopep python=3.10
conda activate geopep
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up HuggingFace token

The ESM3 model requires HuggingFace authentication:

1. Create an account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Create a new token with read access
4. Enable access to gated repositories in your token settings
5. Add your token to `configs/config.yaml`:

```yaml
huggingface:
  token: "hf_your_token_here"
```

---

## Project Structure

```
GeoPep/
├── geopep/                    # Main package
│   ├── models/
│   │   └── esm3_kan.py        # ESM3 + KAN model definition
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset class
│   │   └── __init__.py
│   └── __init__.py
├── model_weights/             # Trained model checkpoints
│   └── model-epoch=XX.ckpt
├── configs/
│   └── config.yaml            # Single config file for everything
├── scripts/
│   ├── preprocess.py          # Step 1: PDB → JSON
│   ├── train.py               # Step 2: Training
│   ├── predict_esm3.py        # Step 3: Prediction
│   └── postprocess.py         # Step 4: Postprocessing
├── pdb/                       # Input PDB files
│   ├── complex/               # Full peptide-protein structures
│   └── interface/             # Interface residues only
├── json/                      # Preprocessed JSON files
│   └── preprocessed/
└── requirements.txt
```

---

## Configuration

All settings are controlled by **one file**: `configs/config.yaml`

### Example Configuration

```yaml
# =============================================================================
# GeoPep Configuration File
# =============================================================================

# -----------------------------------------------------------------------------
# Preprocessing Settings (PDB → JSON)
# -----------------------------------------------------------------------------
preprocess:
  complex_directory: "E:/path/to/pdb/complex"
  interface_directory: "E:/path/to/pdb/interface"
  output_directory: "E:/path/to/json/preprocessed"
  num_json_files: 5

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
data:
  train_json:
    - "E:/path/to/json/preprocessed/data_part_1.json"
    - "E:/path/to/json/preprocessed/data_part_2.json"
    - "E:/path/to/json/preprocessed/data_part_3.json"
  val_json:
    - "E:/path/to/json/preprocessed/data_part_4.json"

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
model:
  peptide_len: 50
  protein_len: 500
  num_label_types: 3

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
training:
  batch_size: 4
  learning_rate: 0.0001
  max_epochs: 100
  use_distance_loss: true

# -----------------------------------------------------------------------------
# Prediction Settings
# -----------------------------------------------------------------------------
prediction:
  checkpoint_path: "E:/path/to/model_weights/model-epoch=01.ckpt"
  input_json: "E:/path/to/json/preprocessed/data_part_5.json"
  input_field: "combined_chains"
  device: "cuda"

# -----------------------------------------------------------------------------
# Hardware Settings
# -----------------------------------------------------------------------------
hardware:
  gpus: [0]
  precision: 16

# -----------------------------------------------------------------------------
# HuggingFace Settings
# -----------------------------------------------------------------------------
huggingface:
  token: "hf_your_token_here"
```

---

## Complete Pipeline

### Step 1: Preprocessing (PDB → JSON)

Converts PDB structure files into JSON format with sequences, labels, and distance maps.

#### 1.1 Prepare your PDB files

Organize your PDB files into two directories:

```
pdb/
├── complex/           # Full peptide-protein structures
│   ├── 1abc_A_B.pdb   # Naming: PDBID_PeptideChain_ProteinChain.pdb
│   ├── 1xyz_C_D.pdb
│   └── ...
└── interface/         # Interface residues only (same filenames!)
    ├── 1abc_A_B.pdb
    ├── 1xyz_C_D.pdb
    └── ...
```

**Important:**
- File names must match between `complex/` and `interface/` directories
- Naming convention: `PDBID_PeptideChain_ProteinChain.pdb`
- Peptide chain should be listed first, protein chain second

#### 1.2 Edit config.yaml

```yaml
preprocess:
  complex_directory: "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/pdb/complex"
  interface_directory: "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/pdb/interface"
  output_directory: "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/json/preprocessed"
  num_json_files: 5
```

#### 1.3 Run preprocessing

```bash
cd E:\Research\2025\drugDesign\code\github\GeoPep\GeoPep\scripts
python preprocess.py
```

Or with explicit config path:

```bash
python preprocess.py --config ../configs/config.yaml
```

#### 1.4 Output

Creates JSON files in the output directory:
- `data_part_1.json`, `data_part_2.json`, ..., `data_part_5.json`
- `error_log.txt` (if any errors occurred)

Each JSON file contains:
```json
{
  "PDBID": {
    "combined_chains": {
      "A_B": "PEPTIDESEQ<pad>...|PROTEINSEQ<pad>..."
    },
    "posIdx_binary": {
      "A_B": "0 1 1 0 2 2 ... 3 0 0 1 1 0 2 2 ..."
    },
    "distance": {
      "A_B": [0, 5.2, 3.1, -2, -2, ..., -1, 0, 2.5, ...]
    }
  }
}
```

---

### Step 2: Training

Train the ESM3-KAN model on preprocessed data.

#### 2.1 Edit config.yaml

```yaml
data:
  train_json:
    - "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/json/preprocessed/data_part_1.json"
    - "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/json/preprocessed/data_part_2.json"
    - "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/json/preprocessed/data_part_3.json"
  val_json:
    - "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/json/preprocessed/data_part_4.json"

training:
  batch_size: 4          # Reduce if GPU memory is limited
  learning_rate: 0.0001
  max_epochs: 100        # Number of training epochs
  use_distance_loss: true  # Enable geometric distance loss

hardware:
  gpus: [0]              # GPU device ID
  precision: 16          # Mixed precision (16 or 32)
```

#### 2.2 Run training

```bash
cd E:\Research\2025\drugDesign\code\github\GeoPep\GeoPep\scripts
python train.py
```

Or with explicit config path:

```bash
python train.py --config ../configs/config.yaml
```

#### 2.3 Output

Trained model checkpoints are saved to `model_weights/`:
- `model-epoch=00.ckpt`
- `model-epoch=01.ckpt`
- etc.

Training progress is displayed:
```
Epoch 0: 100%|████████| 10/10 [03:44<00:00, train_loss=2.930, val_loss=3.200]
```

#### 2.4 Training Tips

- **GPU Memory**: If you get CUDA out of memory errors, reduce `batch_size`
- **Training Time**: Each epoch takes ~4 hours on RTX 3090 with small dataset
- **Checkpoints**: Models are saved every epoch; only best model is kept

---

### Step 3: Prediction

Run inference on new data using a trained model.

#### 3.1 Edit config.yaml

```yaml
prediction:
  # Path to trained model checkpoint
  checkpoint_path: "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/model_weights/model-epoch=01.ckpt"

  # Input JSON file (from preprocessing or custom)
  input_json: "E:/Research/2025/drugDesign/code/github/GeoPep/GeoPep/json/preprocessed/data_part_5.json"

  # Field name containing input sequences
  input_field: "combined_chains"

  # Device: "cuda" or "cpu"
  device: "cuda"
```

#### 3.2 Run prediction

```bash
cd E:\Research\2025\drugDesign\code\github\GeoPep\GeoPep\scripts
python predict_esm3.py
```

Or with explicit config path:

```bash
python predict_esm3.py --config ../configs/config.yaml
```

#### 3.3 Output

Predictions are added to the input JSON file:

```json
{
  "PDBID": {
    "combined_chains": {
      "A_B": "PEPTIDESEQ<pad>...|PROTEINSEQ<pad>..."
    },
    "posIdx_binary": {
      "A_B": "0 1 1 0 ..."
    },
    "model_out_argmax": {
      "A_B": [0, 1, 1, 0, 2, 2, ..., 0, 0, 1, 1, 0, 2, 2, ...]
    },
    "model_out_softmax": {
      "A_B": [[[0.8, 0.15, 0.05], [0.2, 0.7, 0.1], ...]]
    }
  }
}
```

**Output fields:**
- `model_out_argmax`: Predicted class for each position (0, 1, or 2)
- `model_out_softmax`: Probability scores for each class at each position

---

### Step 4: Postprocessing

Converts 3-class predictions to 2-class binary probabilities.

#### 4.1 Run postprocessing

```bash
cd E:\Research\2025\drugDesign\code\github\GeoPep\GeoPep\scripts
python postprocess.py --input E:/path/to/predictions.json
```

#### 4.2 Output

Adds clean binary probabilities to the JSON:

```json
{
  "PDBID": {
    "peptide_out": [[0.8, 0.2], [0.2, 0.8], ...],
    "protein_out": [[0.9, 0.1], [0.3, 0.7], ...]
  }
}
```

---

## Data Format

### Input Sequence Format

```
PEPTIDESEQ<pad><pad>...|PROTEINSEQ<pad><pad>...
|<----- 50 chars ----->|<------ 500 chars ----->|
```

- Peptide: Maximum 50 residues, padded with `<pad>`
- Protein: Maximum 500 residues, padded with `<pad>`
- Separator: `|` character between peptide and protein

### Label Format

| Value | Meaning |
|-------|---------|
| 0 | Non-interface residue |
| 1 | Interface (binding site) residue |
| 2 | Padding position |
| 3 | Separator (between peptide and protein) |

### Output Positions

```
Position:  0 -------- 49   50    51 -------- 550
Content:   [  peptide  ]  [sep]  [   protein   ]
Label:     [0,1,2 vals ]  [ 3 ]  [ 0,1,2 vals  ]
```

### Distance Format

- `0`: Interface residue (distance to interface = 0)
- `>0`: Non-interface residue (normalized distance to nearest interface, 0-10 scale)
- `-1`: Separator position
- `-2`: Padding position

---

## Troubleshooting

### Common Errors

#### 1. HuggingFace Authentication Error
```
401 Client Error: Unauthorized
```
**Solution:**
- Check your HuggingFace token in `config.yaml`
- Enable access to gated repositories in your HuggingFace settings

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce `batch_size` in `config.yaml`
- Use `precision: 16` for mixed precision training

#### 3. Checkpoint File Corrupted
```
RuntimeError: PytorchStreamReader failed reading zip archive
```
**Solution:**
- The checkpoint file is corrupted (likely from interrupted training)
- Use a different checkpoint file or retrain

#### 4. CUDA Assertion Error
```
CUDA error: device-side assert triggered
```
**Solution:**
- Check that `num_label_types: 3` in config.yaml
- Labels should only contain values 0, 1, 2 (not 3) when computing loss

#### 5. Unicode Decode Error (Windows)
```
UnicodeDecodeError: 'gbk' codec can't decode byte
```
**Solution:**
- Already fixed in code with `encoding='utf-8'` parameter

---

## Quick Reference Commands

```bash
# Navigate to scripts directory
cd E:\Research\2025\drugDesign\code\github\GeoPep\GeoPep\scripts

# Step 1: Preprocess PDB files to JSON
python preprocess.py

# Step 2: Train the model
python train.py

# Step 3: Run predictions
python predict_esm3.py

# Step 4: Postprocess results
python postprocess.py --input ../json/preprocessed/data_part_5.json
```

---

## License

MIT License
