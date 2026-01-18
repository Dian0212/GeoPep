"""
Preprocessing: Convert PDB files to JSON format for training.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from Bio.PDB import PDBParser
except ImportError:
    PDBParser = None
    print("Warning: BioPython not installed. PDB parsing disabled.")


# Amino acid conversion
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06
}


def three_to_one_letter(three_letter: str) -> str:
    """Convert three-letter amino acid code to one-letter."""
    return THREE_TO_ONE.get(three_letter, '<unk>')


def calculate_mass_center(residue) -> Optional[np.ndarray]:
    """Calculate center of mass for a residue."""
    total_mass = 0.0
    weighted_coords = np.zeros(3)
    
    for atom in residue.get_atoms():
        element = atom.element.capitalize()
        mass = ATOMIC_MASSES.get(element, 12.011)
        coord = atom.get_coord()
        weighted_coords += mass * coord
        total_mass += mass
    
    if total_mass == 0:
        return None
    return weighted_coords / total_mass


def preprocess_pdb_to_json(
    complex_directory: str,
    interface_directory: str,
    output_directory: str,
    num_json_files: int = 5,
    pad_length: List[int] = [50, 500],
    length_threshold: List[int] = [10, 10]
) -> None:
    """
    Convert PDB files to JSON format for training.
    
    Args:
        complex_directory: Directory with whole complex PDB files
        interface_directory: Directory with interface PDB files
        output_directory: Output directory for JSON files
        num_json_files: Number of output JSON files
        pad_length: [peptide_len, protein_len] for padding
        length_threshold: Minimum [peptide_len, protein_len]
    """
    if PDBParser is None:
        raise ImportError("BioPython required for PDB parsing")
    
    os.makedirs(output_directory, exist_ok=True)
    parser = PDBParser(QUIET=True)
    
    # Get all PDB files
    pdb_files = [f for f in os.listdir(complex_directory) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files")
    
    # Split into groups
    chunk_size = len(pdb_files) // num_json_files + 1
    groups = [pdb_files[i:i+chunk_size] for i in range(0, len(pdb_files), chunk_size)]
    
    for group_idx, pdb_group in enumerate(groups):
        json_data = {}
        
        for pdb_file in pdb_group:
            # Parse PDB and extract data
            # (Implementation depends on your PDB format)
            pass
        
        # Save JSON
        output_path = os.path.join(output_directory, f"data_part_{group_idx+1}.json")
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"Saved {output_path}")
