#!/usr/bin/env python3
"""
GeoPep One-Command Inference Pipeline

Given a folder of PDB files and a trained model checkpoint, run the full
preprocessing -> prediction -> result-JSON pipeline. The final output is
result/predictions.json with peptide/protein sequences and per-residue
binding probabilities (padding stripped).

Usage:
    python inference_pipeline.py --pdb-dir /path/to/pdb --checkpoint /path/to/model.ckpt
    python inference_pipeline.py --pdb-dir /path/to/pdb --checkpoint /path/to/model.ckpt --result-dir /path/to/result

PDB folder layout (either flat or with a `complex/` subfolder is accepted):
    /path/to/pdb/
        1abc_A_B.pdb        # Naming: PDBID_PeptideChain_ProteinChain.pdb
        1xyz_C_D.pdb
        ...
    OR
    /path/to/pdb/
        complex/
            1abc_A_B.pdb
            ...

NOTE: Inference does NOT need interface labels. If an interface/ subfolder
exists alongside complex/, it is ignored.
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

from preprocess import run_preprocessing
from predict_esm3 import run_prediction
from postprocess import process_predictions, save_result_json


def _resolve_complex_dir(pdb_dir: str) -> str:
    """Accept either a flat PDB folder or one with a complex/ subdir."""
    sub = os.path.join(pdb_dir, "complex")
    if os.path.isdir(sub):
        return sub
    return pdb_dir


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end GeoPep inference: PDB folder + checkpoint -> result/predictions.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        required=True,
        help="Folder containing the PDB files to predict on (flat, or with a complex/ subdirectory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model .ckpt file.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="../result",
        help="Directory to save predictions.json (default: ../result).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/config.yaml",
        help="Base config file (HF token, device, model dims read from here). Default: ../configs/config.yaml.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="../json/inference",
        help="Where to write the intermediate preprocessed JSON (default: ../json/inference).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cuda", "cpu"),
        default=None,
        help="Override prediction device (default: read from config, usually cuda).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing if --work-dir already contains data_part_1.json.",
    )
    args = parser.parse_args()

    pdb_dir = os.path.abspath(args.pdb_dir)
    checkpoint = os.path.abspath(args.checkpoint)
    result_dir = os.path.abspath(args.result_dir)
    work_dir = os.path.abspath(args.work_dir)
    config_path = os.path.abspath(args.config)

    if not os.path.isdir(pdb_dir):
        sys.exit(f"ERROR: --pdb-dir not found: {pdb_dir}")
    if not os.path.isfile(checkpoint):
        sys.exit(f"ERROR: --checkpoint not found: {checkpoint}")
    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: --config not found: {config_path}")

    complex_dir = _resolve_complex_dir(pdb_dir)

    print("=" * 70)
    print("GeoPep Inference Pipeline")
    print("=" * 70)
    print(f"PDB dir:         {pdb_dir}")
    print(f"  complex src:   {complex_dir}")
    print(f"Checkpoint:      {checkpoint}")
    print(f"Work dir:        {work_dir}")
    print(f"Result dir:      {result_dir}")
    print(f"Config:          {config_path}")
    print("=" * 70)

    # --- Load base config ---
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Override preprocess section (inference-only: no interface dir) ---
    config.setdefault("preprocess", {})
    config["preprocess"]["complex_directory"] = complex_dir
    config["preprocess"]["interface_directory"] = None  # triggers inference-only mode
    config["preprocess"]["output_directory"] = work_dir
    config["preprocess"]["num_json_files"] = 1  # single file for simpler inference

    # --- Step 1: Preprocess ---
    intermediate_json = os.path.join(work_dir, "data_part_1.json")

    if args.skip_preprocess and os.path.isfile(intermediate_json):
        print(f"\n[1/3] Skipping preprocessing — found {intermediate_json}")
    else:
        print(f"\n[1/3] Preprocessing PDBs ...")
        produced = run_preprocessing(config)
        if not produced or not os.path.isfile(intermediate_json):
            sys.exit("ERROR: preprocessing produced no JSON file (no valid PDBs?).")

    # --- Step 2: Prediction ---
    config.setdefault("prediction", {})
    config["prediction"]["checkpoint_path"] = checkpoint
    config["prediction"]["input_json"] = intermediate_json
    config["prediction"].setdefault("input_field", "combined_chains")
    if args.device is not None:
        config["prediction"]["device"] = args.device
    else:
        config["prediction"].setdefault("device", "cuda")

    print(f"\n[2/3] Running prediction ...")
    run_prediction(config)

    # --- Step 3: Build result JSON ---
    print(f"\n[3/3] Building result JSON ...")
    os.makedirs(result_dir, exist_ok=True)
    result_accumulator = {}
    process_predictions(intermediate_json, result_accumulator=result_accumulator)
    out_path = save_result_json(result_accumulator, result_dir)

    print("\n" + "=" * 70)
    print("Inference pipeline complete.")
    print(f"Result file:    {out_path}")
    print(f"Total entries:  {len(result_accumulator)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
