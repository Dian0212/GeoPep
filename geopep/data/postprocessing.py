"""
Postprocessing: Convert model predictions to binary probabilities.
"""

import json
import ast
from typing import List, Tuple


def process_softmax_to_binary(softmax_probs: List[List[float]]) -> List[List[float]]:
    """
    Convert 3-class softmax to 2-class binary probabilities.
    
    Combines class 0 (non-interface) and class 2 (padding) as negative.
    Class 1 (interface) is positive.
    
    Args:
        softmax_probs: List of [prob_0, prob_1, prob_2] per position
        
    Returns:
        List of [prob_negative, prob_positive] per position
    """
    binary_probs = []
    for prob_0, prob_1, prob_2 in softmax_probs:
        combined_neg = prob_0 + prob_2
        binary_probs.append([combined_neg, prob_1])
    return binary_probs


def process_prediction_file(
    json_file_path: str,
    peptide_out_field: str = "peptide_out",
    protein_out_field: str = "protein_out",
    peptide_start: int = 0,
    protein_start: int = 51
) -> None:
    """
    Process predictions to extract peptide and protein binary probabilities.
    
    Args:
        json_file_path: Path to JSON with predictions
        peptide_out_field: Output field for peptide predictions
        protein_out_field: Output field for protein predictions
        peptide_start: Start position for peptide
        protein_start: Start position for protein
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    
    for entry_key, entry_data in data.items():
        try:
            # Parse softmax output
            softmax_str = entry_data.get("model_out_softmax")
            if not softmax_str:
                continue
            
            if isinstance(softmax_str, str):
                softmax_output = ast.literal_eval(softmax_str)
            else:
                softmax_output = softmax_str
            
            # Handle batch dimension
            predictions = softmax_output[0] if len(softmax_output) == 1 else softmax_output
            
            # Get chain IDs for lengths
            parts = entry_key.split('_')
            if len(parts) >= 3:
                pep_chain = parts[1]
                prot_chain = parts[2]
                pep_len = len(entry_data.get(pep_chain, "")) or 50
                prot_len = len(entry_data.get(prot_chain, "")) or 500
            else:
                pep_len, prot_len = 50, 500
            
            # Extract and convert predictions
            pep_softmax = predictions[peptide_start:peptide_start + pep_len]
            prot_softmax = predictions[protein_start:protein_start + prot_len]
            
            entry_data[peptide_out_field] = process_softmax_to_binary(pep_softmax)
            entry_data[protein_out_field] = process_softmax_to_binary(prot_softmax)
            
        except Exception as e:
            print(f"Error processing {entry_key}: {e}")
    
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {json_file_path}")
