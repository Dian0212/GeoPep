#!/usr/bin/env python3
"""
GeoPep One-Command Training Pipeline

Given a PDB folder containing `complex/` and `interface/` subdirectories,
preprocess all PDBs to JSON and train the per-residue ESM3-KAN model.
The trained checkpoint is saved to --output-dir (default: ../model_weights).

Usage:
    python train_pipeline.py --pdb-dir /path/to/pdb
    python train_pipeline.py --pdb-dir /path/to/pdb --output-dir /path/to/save
    python train_pipeline.py --pdb-dir /path/to/pdb --config ../configs/config.yaml

Expected PDB folder layout:
    /path/to/pdb/
        complex/      # Full peptide-protein complex structures
            1abc_A_B.pdb       # Naming: PDBID_PeptideChain_ProteinChain.pdb
            ...
        interface/    # Interface residues only (same filenames as complex/)
            1abc_A_B.pdb
            ...
"""

import argparse
import os
import sys
import yaml

# Force UTF-8 stdout/stderr so emoji prints don't crash on Windows (gbk codec).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from preprocess import run_preprocessing
from train import run_training
from geopep.config_utils import resolve_paths


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end GeoPep training pipeline: PDB folder -> trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        required=True,
        help="Root PDB folder containing 'complex/' and 'interface/' subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../model_weights",
        help="Directory to save the trained checkpoint (default: ../model_weights).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/config.yaml",
        help="Base config file. CLI flags override its fields where applicable (default: ../configs/config.yaml).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="../json/preprocessed",
        help="Where to write preprocessed JSON files (default: ../json/preprocessed).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of preprocessed JSON files used as validation set (default: 0.2).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing if --work-dir already contains data_part_*.json files.",
    )
    args = parser.parse_args()

    # --- Validate PDB folder layout ---
    pdb_dir = os.path.abspath(args.pdb_dir)
    complex_dir = os.path.join(pdb_dir, "complex")
    interface_dir = os.path.join(pdb_dir, "interface")

    if not os.path.isdir(complex_dir):
        sys.exit(f"ERROR: missing subdirectory: {complex_dir}")
    if not os.path.isdir(interface_dir):
        sys.exit(f"ERROR: missing subdirectory: {interface_dir}")

    work_dir = os.path.abspath(args.work_dir)
    output_dir = os.path.abspath(args.output_dir)
    config_path = os.path.abspath(args.config)

    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: config file not found: {config_path}")

    print("=" * 70)
    print("GeoPep Training Pipeline")
    print("=" * 70)
    print(f"PDB dir:         {pdb_dir}")
    print(f"  complex/:      {complex_dir}")
    print(f"  interface/:    {interface_dir}")
    print(f"Work dir:        {work_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Config:          {config_path}")
    print("=" * 70)

    # --- Load base config and apply overrides ---
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    resolve_paths(config, config_path)

    config.setdefault("preprocess", {})
    config["preprocess"]["complex_directory"] = complex_dir
    config["preprocess"]["interface_directory"] = interface_dir
    config["preprocess"]["output_directory"] = work_dir

    num_json_files = config["preprocess"].get("num_json_files", 5)
    if num_json_files < 2:
        sys.exit("ERROR: preprocess.num_json_files must be >= 2 (need at least 1 train + 1 val).")

    # --- Step 1: Preprocess PDB -> JSON ---
    expected_files = [
        os.path.join(work_dir, f"data_part_{i + 1}.json")
        for i in range(num_json_files)
    ]
    have_all = all(os.path.isfile(p) for p in expected_files)

    if args.skip_preprocess and have_all:
        print("\n[1/2] Skipping preprocessing — JSON files already exist.")
        for p in expected_files:
            print(f"  found: {p}")
        produced_files = expected_files
    else:
        print(f"\n[1/2] Preprocessing PDB files into {num_json_files} JSON chunks ...")
        produced_files = run_preprocessing(config)

    if not produced_files:
        sys.exit("ERROR: preprocessing produced no JSON files (no valid PDBs?).")

    # --- Choose train/val split from produced files ---
    n_total = len(produced_files)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_val = min(n_val, n_total - 1)  # always keep at least 1 train file
    train_files = produced_files[: n_total - n_val]
    val_files = produced_files[n_total - n_val:]

    config.setdefault("data", {})
    config["data"]["train_json"] = train_files
    config["data"]["val_json"] = val_files

    print(f"\nSplit: {len(train_files)} train file(s) + {len(val_files)} val file(s)")

    # --- Step 2: Train ---
    config.setdefault("training", {})
    config["training"]["checkpoint_dir"] = output_dir

    print(f"\n[2/2] Training model ...")
    best_ckpt = run_training(config)

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print(f"Best checkpoint: {best_ckpt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
