#!/usr/bin/env python3
"""
GeoPep Postprocessing Script

Convert 3-class softmax to 2-class binary probabilities.

Usage: python postprocess.py --input predictions.json
"""

import json
import ast
import argparse


def process_softmax_to_binary(softmax_probs):
    """Convert 3-class to 2-class: combine class 0+2 vs class 1."""
    binary_probs = []
    for prob_0, prob_1, prob_2 in softmax_probs:
        combined_neg = prob_0 + prob_2
        binary_probs.append([combined_neg, prob_1])
    return binary_probs


def process_predictions(json_file_path, peptide_out="peptide_out", protein_out="protein_out"):
    """Process all predictions in JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    processed = 0
    
    for entry_key, entry_data in data.items():
        try:
            softmax_str = entry_data.get("model_out_softmax")
            if not softmax_str:
                continue
            
            if isinstance(softmax_str, str):
                softmax_output = ast.literal_eval(softmax_str)
            else:
                softmax_output = softmax_str
            
            predictions = softmax_output[0] if len(softmax_output) == 1 else softmax_output
            
            # Get lengths from entry key
            parts = entry_key.split('_')
            if len(parts) >= 3:
                pep_chain = parts[1]
                prot_chain = parts[2]
                pep_len = len(entry_data.get(pep_chain, "")) or 50
                prot_len = len(entry_data.get(prot_chain, "")) or 500
            else:
                pep_len, prot_len = 50, 500
            
            # Extract and convert
            pep_softmax = predictions[0:pep_len]
            prot_softmax = predictions[51:51 + prot_len]
            
            entry_data[peptide_out] = process_softmax_to_binary(pep_softmax)
            entry_data[protein_out] = process_softmax_to_binary(prot_softmax)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {entry_key}: {e}")
    
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Processed {processed} entries, saved to {json_file_path}")


def calculate_metrics(json_file_path, threshold=0.5):
    """Calculate prediction metrics."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for entry_key, entry_data in data.items():
        if "protein_out" not in entry_data:
            continue
        
        parts = entry_key.split('_')
        if len(parts) >= 3:
            prot_chain = parts[2]
            if prot_chain in entry_data:
                gt = entry_data[prot_chain]
                preds = entry_data["protein_out"]
                
                for gt_char, pred_probs in zip(gt, preds):
                    gt_val = int(gt_char)
                    pred_val = 1 if pred_probs[1] >= threshold else 0
                    
                    if gt_val == 1 and pred_val == 1: tp += 1
                    elif gt_val == 0 and pred_val == 1: fp += 1
                    elif gt_val == 1 and pred_val == 0: fn += 1
                    else: tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics (threshold={threshold}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")


def main():
    parser = argparse.ArgumentParser(description="Postprocess predictions")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--calculate-metrics", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    process_predictions(args.input)
    
    if args.calculate_metrics:
        calculate_metrics(args.input, args.threshold)


if __name__ == "__main__":
    main()
