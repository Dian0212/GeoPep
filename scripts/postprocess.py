#!/usr/bin/env python3
"""
GeoPep Postprocessing Script

Convert 3-class softmax to 2-class binary probabilities and generate CSV files.

Usage:
    python postprocess.py --input predictions.json
    python postprocess.py --input-dir /path/to/json/folder
    python postprocess.py --input predictions.json --output-dir /path/to/csv/output
"""

import json
import ast
import argparse
import os
import csv


def process_softmax_to_binary(softmax_probs):
    """Convert 3-class to 2-class: combine class 0+2 vs class 1.

    Input format from model: (3, num_positions) - 3 class probabilities for each position
    Output format: (num_positions, 2) - binary probabilities for each position
    """
    binary_probs = []
    # softmax_probs shape is (3, num_positions): [class0_probs, class1_probs, class2_probs]
    class0_probs = softmax_probs[0]  # Non-interface
    class1_probs = softmax_probs[1]  # Interface
    class2_probs = softmax_probs[2]  # Padding

    num_positions = len(class0_probs)
    for i in range(num_positions):
        prob_0 = class0_probs[i]
        prob_1 = class1_probs[i]
        prob_2 = class2_probs[i]
        combined_neg = prob_0 + prob_2
        binary_probs.append([combined_neg, prob_1])
    return binary_probs


def extract_sequences(combined_chain_str):
    """Extract peptide and protein sequences from combined chain string.

    Format: PEPTIDESEQ<pad>...|PROTEINSEQ<pad>...
    """
    # Split by separator '|'
    parts = combined_chain_str.split('|')
    if len(parts) != 2:
        return None, None

    peptide_part = parts[0]  # First 50 characters (peptide + padding)
    protein_part = parts[1]  # Next 500 characters (protein + padding)

    # Extract actual residues (remove <pad>, <unk> tokens)
    peptide_seq = peptide_part.replace('<pad>', '').replace('<unk>', 'X')
    protein_seq = protein_part.replace('<pad>', '').replace('<unk>', 'X')

    return peptide_seq, protein_seq


def get_binding_prediction(binary_probs, threshold=0.5):
    """Convert binary probabilities to binding prediction (0 or 1).

    binary_probs: list of [non_binding_prob, binding_prob]
    Returns: 1 if binding_prob > threshold, else 0
    """
    return 1 if binary_probs[1] > threshold else 0


def generate_csv_for_sample(pdb_id, chain_key, peptide_seq, protein_seq,
                            peptide_probs, protein_probs, output_dir, threshold=0.5):
    """Generate CSV files for a single sample.

    Creates two CSV files:
    - {pdb_id}_{chain_key}_peptide.csv
    - {pdb_id}_{chain_key}_protein.csv

    Each CSV has columns: Position, Residue, Binding (0 or 1), Binding_Probability
    Only includes actual residues (skips padding positions).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate peptide CSV (only actual residues, no padding)
    peptide_csv_path = os.path.join(output_dir, f"{pdb_id}_{chain_key}_peptide.csv")
    with open(peptide_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Position', 'Residue', 'Binding', 'Binding_Probability'])

        for i in range(len(peptide_seq)):
            residue = peptide_seq[i]
            probs = peptide_probs[i]
            binding = get_binding_prediction(probs, threshold)
            binding_prob = probs[1]  # Probability of binding
            writer.writerow([i + 1, residue, binding, f"{binding_prob:.4f}"])

    # Generate protein CSV (only actual residues, no padding)
    protein_csv_path = os.path.join(output_dir, f"{pdb_id}_{chain_key}_protein.csv")
    with open(protein_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Position', 'Residue', 'Binding', 'Binding_Probability'])

        for i in range(len(protein_seq)):
            residue = protein_seq[i]
            probs = protein_probs[i]
            binding = get_binding_prediction(probs, threshold)
            binding_prob = probs[1]  # Probability of binding
            writer.writerow([i + 1, residue, binding, f"{binding_prob:.4f}"])

    return peptide_csv_path, protein_csv_path


def process_predictions(json_file_path, output_dir=None, threshold=0.5):
    """Process all predictions in JSON file and optionally generate CSV files."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} PDB entries in {os.path.basename(json_file_path)}...")
    processed = 0
    csv_generated = 0

    for pdb_id, pdb_data in data.items():
        # Handle nested structure: model_out_softmax is a dict with chain keys
        if "model_out_softmax" in pdb_data and isinstance(pdb_data["model_out_softmax"], dict):
            # Initialize output dicts
            if "peptide_out" not in pdb_data:
                pdb_data["peptide_out"] = {}
            if "protein_out" not in pdb_data:
                pdb_data["protein_out"] = {}

            for chain_key, softmax_output in pdb_data["model_out_softmax"].items():
                try:
                    # Handle string or list format
                    if isinstance(softmax_output, str):
                        softmax_output = ast.literal_eval(softmax_output)

                    # Get predictions - shape is (1, 3, 551) from model
                    # softmax_output[0] gives shape (3, 551): [class0_probs, class1_probs, class2_probs]
                    predictions = softmax_output[0] if len(softmax_output) == 1 else softmax_output

                    # predictions is now shape (3, 551)
                    # Extract peptide (positions 0-49) and protein (positions 51-550)
                    # For each class, slice the positions
                    pep_softmax = [predictions[c][0:50] for c in range(3)]  # Shape (3, 50)
                    prot_softmax = [predictions[c][51:551] for c in range(3)]  # Shape (3, 500)

                    # Convert to binary
                    peptide_probs = process_softmax_to_binary(pep_softmax)
                    protein_probs = process_softmax_to_binary(prot_softmax)

                    pdb_data["peptide_out"][chain_key] = peptide_probs
                    pdb_data["protein_out"][chain_key] = protein_probs
                    processed += 1

                    # Generate CSV files if output_dir is specified
                    if output_dir:
                        # Get sequences from combined_chains
                        combined_str = pdb_data.get("combined_chains", {}).get(chain_key, "")
                        peptide_seq, protein_seq = extract_sequences(combined_str)

                        if peptide_seq and protein_seq:
                            generate_csv_for_sample(
                                pdb_id, chain_key, peptide_seq, protein_seq,
                                peptide_probs, protein_probs, output_dir, threshold
                            )
                            csv_generated += 1

                except Exception as e:
                    print(f"  Error processing {pdb_id}_{chain_key}: {e}")

        # Handle flat structure (single softmax per entry)
        elif "model_out_softmax" in pdb_data:
            try:
                softmax_output = pdb_data["model_out_softmax"]
                if isinstance(softmax_output, str):
                    softmax_output = ast.literal_eval(softmax_output)

                # Get predictions - shape is (1, 3, 551) from model
                predictions = softmax_output[0] if len(softmax_output) == 1 else softmax_output

                # predictions is now shape (3, 551)
                pep_softmax = [predictions[c][0:50] for c in range(3)]  # Shape (3, 50)
                prot_softmax = [predictions[c][51:551] for c in range(3)]  # Shape (3, 500)

                peptide_probs = process_softmax_to_binary(pep_softmax)
                protein_probs = process_softmax_to_binary(prot_softmax)

                pdb_data["peptide_out"] = peptide_probs
                pdb_data["protein_out"] = protein_probs
                processed += 1

                # Generate CSV files if output_dir is specified
                if output_dir:
                    combined_str = pdb_data.get("combined_chains", "")
                    if isinstance(combined_str, str):
                        peptide_seq, protein_seq = extract_sequences(combined_str)
                        if peptide_seq and protein_seq:
                            generate_csv_for_sample(
                                pdb_id, "default", peptide_seq, protein_seq,
                                peptide_probs, protein_probs, output_dir, threshold
                            )
                            csv_generated += 1

            except Exception as e:
                print(f"  Error processing {pdb_id}: {e}")

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"  Processed {processed} samples, saved to {json_file_path}")
    if output_dir:
        print(f"  Generated {csv_generated * 2} CSV files in {output_dir}")
    return processed, csv_generated


def process_directory(input_dir, output_dir=None, threshold=0.5):
    """Process all JSON files in a directory."""
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to process\n")

    total_processed = 0
    total_csv = 0
    for json_file in sorted(json_files):
        json_path = os.path.join(input_dir, json_file)
        processed, csv_count = process_predictions(json_path, output_dir, threshold)
        total_processed += processed
        total_csv += csv_count

    print(f"\n{'='*60}")
    print(f"TOTAL: Processed {total_processed} samples across {len(json_files)} files")
    if output_dir:
        print(f"TOTAL: Generated {total_csv * 2} CSV files in {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Postprocess predictions and generate CSV files")
    parser.add_argument("--input", type=str, help="Single JSON file to process")
    parser.add_argument("--input-dir", type=str, help="Directory containing JSON files to process")
    parser.add_argument("--output-dir", type=str, help="Directory to save CSV files (if not specified, no CSV files are generated)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binding prediction (default: 0.5)")
    args = parser.parse_args()

    if args.input_dir:
        process_directory(args.input_dir, args.output_dir, args.threshold)
    elif args.input:
        process_predictions(args.input, args.output_dir, args.threshold)
    else:
        print("Please provide --input or --input-dir")
        print("Example:")
        print("  python postprocess.py --input predictions.json")
        print("  python postprocess.py --input-dir /path/to/json/folder")
        print("  python postprocess.py --input predictions.json --output-dir /path/to/csv/output")
        print("  python postprocess.py --input-dir /path/to/json --output-dir /path/to/csv --threshold 0.6")


if __name__ == "__main__":
    main()
