#!/usr/bin/env python3
"""
GeoPep Preprocessing Script

Convert PDB files to JSON format for training.
Based on parseJson_esm3.py and calculateDistanceMap_1D.py reference implementations.

Usage:
    python preprocess.py --config ../configs/config.yaml
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
from Bio.PDB import PDBParser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from geopep.config_utils import resolve_paths

# Atomic masses for common elements (in atomic mass units, amu)
atomic_masses = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'S': 32.06, 'Cl': 35.45, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956,
    'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845,
    'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723,
    'Ge': 72.63, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798
}

# Function to convert three letter residue name to one letter
def three_to_one_letter(three_letter_residue):
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "SEC": "U", "PYL": "O", "ORN": "O"
    }
    return three_to_one.get(three_letter_residue, '<unk>')


def get_atomic_mass(atom):
    """Get the atomic mass of an atom using its element."""
    element = atom.element.capitalize()
    return atomic_masses.get(element, 12.011)  # Default to carbon


def calculate_mass_center(residue):
    """Calculate the mass-weighted center of mass (COM) for a residue."""
    total_mass = 0.0
    weighted_coords = np.zeros(3)

    for atom in residue.get_atoms():
        mass = get_atomic_mass(atom)
        coord = atom.get_coord()
        weighted_coords += mass * coord
        total_mass += mass

    if total_mass == 0:
        return None
    return weighted_coords / total_mass


def calculate_distances_com(chain, interface_binary, pdb_info, error_log=None):
    """
    Calculate distances based on center of mass (COM) between
    interface and non-interface residues.
    """
    distances = []
    residues = list(chain.get_residues())

    # Calculate mass-weighted COM for all residues
    residues_com = [calculate_mass_center(residue) for residue in residues]

    # Get COMs for interface residues (where label == 1)
    interface_coms = []
    for i, is_interface in enumerate(interface_binary):
        if is_interface == 1 and i < len(residues_com) and residues_com[i] is not None:
            interface_coms.append(residues_com[i])

    if len(interface_coms) == 0:
        if error_log:
            error_log.write(f"No interface residues found for {pdb_info}. Skipping distance calculation.\n")
        return [-1] * len(residues)

    interface_coms = np.array(interface_coms)

    for i, is_interface in enumerate(interface_binary):
        if i >= len(residues_com):
            distances.append(-1)
            continue

        if is_interface == 1:
            distances.append(0)
        else:
            com = residues_com[i]
            if com is not None:
                com_distances = np.linalg.norm(interface_coms - com, axis=1)
                min_distance = np.min(com_distances)
                distances.append(min_distance)
            else:
                distances.append(-1)

    return distances


def pad_list(input_list, target_length, padding_value):
    """Pad a list to target length."""
    if len(input_list) > target_length:
        return input_list[:target_length]
    return input_list + [padding_value] * (target_length - len(input_list))


def extract_sequence_posIdx(pdb_path, chains):
    sequences = {}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure_id', pdb_path)

        model_chains = {chain.id for model in structure for chain in model}
        chain_exist = [chain in model_chains for chain in chains]
        chains = [chain.swapcase() if not exist else chain for chain, exist in zip(chains, chain_exist)]

        for idx, chain_idIndividual_chains in enumerate(chains):
            for chain in structure[0]:
                if chain.id == chain_idIndividual_chains:
                    chain_id = chain_idIndividual_chains.swapcase() if not chain_exist[idx] else chain_idIndividual_chains
                    if chain_id not in sequences:
                        sequences[chain_id] = []
                    for residue in chain:
                        residue_index = str(residue.id[1])
                        sequences[chain_id].append(residue_index)
        return {chain: ' '.join(sequences[chain]) for chain in sequences if sequences[chain]}
    except FileNotFoundError:
        print(f"File not found: {pdb_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def extract_sequence_combinedChains(pdb_path, chains, pad_length, length_threshold):
    lenCheck_flag = True
    sequences = {}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure_id', pdb_path)

        model_chains = {chain.id for model in structure for chain in model}
        chain_exist = [chain in model_chains for chain in chains]
        chains = [chain.swapcase() if not exist else chain for chain, exist in zip(chains, chain_exist)]

        for idx, chain_idIndividual_chains in enumerate(chains):
            for chain in structure[0]:
                if chain.id == chain_idIndividual_chains:
                    chain_id = chain_idIndividual_chains.swapcase() if not chain_exist[idx] else chain_idIndividual_chains
                    if chain_id not in sequences:
                        sequences[chain_id] = []
                    for residue in chain:
                        residue_name = three_to_one_letter(residue.resname)
                        sequences[chain_id].append(residue_name)

        for (chain_id, chain_value), length, len_threshold in zip(sequences.items(), pad_length, length_threshold):
            part_list = chain_value
            if len(part_list) > length or len(part_list) < len_threshold:
                lenCheck_flag = False
                return {}, lenCheck_flag
            while len(part_list) < length:
                part_list.append('<pad>')
            sequences[chain_id] = part_list
        return {chain: ''.join(sequences[chain]) for chain in sequences if sequences[chain]}, lenCheck_flag
    except FileNotFoundError:
        print(f"File not found: {pdb_path}")
        return {}, False
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}, False


def generate_binary_string(pdb_id, str_wholeComplex, str_interface, pad_length):
    binary_string = []
    str_wholeComplex_parts = [list(map(int, part.split())) for part in str_wholeComplex.split('|')]
    str_interface_parts = [list(map(int, part.split())) for part in str_interface.split('|')]

    for i1_part, i2_part in zip(str_wholeComplex_parts, str_interface_parts):
        binary_part = ['1' if pos in i2_part else '0' for pos in i1_part]
        binary_string.append(' '.join(binary_part))

    for idx, (part, length) in enumerate(zip(binary_string, pad_length)):
        part_list = part.split()
        num_padding = length - len(part_list)
        if num_padding < 0:
            print("Part length specified is less than the number of elements in the part.")
        padded_part = part_list + ['2'] * num_padding
        binary_string[idx] = ' '.join(padded_part)
    return ' 3 '.join(binary_string)


def calculate_distance_for_entry(pdb_id, combined_chains_key, posIdx_binary_str,
                                  complex_directory_path, pad_length, scale=10, error_log=None):
    """
    Calculate distance map for a single entry.
    Returns combined_distances list or None if error.
    """
    posIdx_binary = list(map(int, posIdx_binary_str.split()))

    # Find separator (value 3)
    try:
        split_index = posIdx_binary.index(3)
    except ValueError:
        if error_log:
            error_log.write(f"No separator found for {pdb_id}_{combined_chains_key}\n")
        return None

    peptide_interface = posIdx_binary[:split_index]
    protein_interface = posIdx_binary[split_index + 1:]

    # Remove padding (value 2) to get actual interface labels
    peptide_interface_clean = [x for x in peptide_interface if x != 2]
    protein_interface_clean = [x for x in protein_interface if x != 2]

    chain_ids = combined_chains_key.split('_')
    if len(chain_ids) != 2:
        if error_log:
            error_log.write(f"Unexpected chain format for PDB ID {pdb_id}: {combined_chains_key}\n")
        return None

    chain1, chain2 = chain_ids
    pdb_file_name = f"{pdb_id}_{combined_chains_key}.pdb"
    pdb_file_path = os.path.join(complex_directory_path, pdb_file_name)

    if not os.path.exists(pdb_file_path):
        if error_log:
            error_log.write(f"PDB file {pdb_file_name} not found\n")
        return None

    try:
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(pdb_id, pdb_file_path)

        # Try to get chains (with case swapping if needed)
        try:
            peptide_chain = structure[0][chain1]
        except KeyError:
            peptide_chain = structure[0][chain1.swapcase()]

        try:
            protein_chain = structure[0][chain2]
        except KeyError:
            protein_chain = structure[0][chain2.swapcase()]

        pdb_info = f"{pdb_id}_{combined_chains_key}"

        # Calculate distances
        peptide_distances = calculate_distances_com(peptide_chain, peptide_interface_clean, pdb_info, error_log)
        protein_distances = calculate_distances_com(protein_chain, protein_interface_clean, pdb_info, error_log)

        # Normalize distances to 0-scale range
        peptide_max_distance = max(peptide_distances) if peptide_distances else 0
        protein_max_distance = max(protein_distances) if protein_distances else 0

        if peptide_max_distance > 0:
            peptide_distances = [scale * d / peptide_max_distance if d >= 0 else d for d in peptide_distances]
        if protein_max_distance > 0:
            protein_distances = [scale * d / protein_max_distance if d >= 0 else d for d in protein_distances]

        # Pad to target lengths
        peptide_distances = pad_list(peptide_distances, target_length=pad_length[0], padding_value=-2)
        protein_distances = pad_list(protein_distances, target_length=pad_length[1], padding_value=-2)

        # Combine with separator
        combined_distances = peptide_distances + [-1] + protein_distances

        return combined_distances

    except Exception as e:
        if error_log:
            error_log.write(f"Error processing {pdb_id}_{combined_chains_key}: {e}\n")
        return None


def run_preprocessing(config):
    """Run PDB → JSON preprocessing given a parsed config dict.

    Returns the list of output JSON file paths produced (data_part_*.json).
    """
    preprocess_cfg = config['preprocess']
    model_cfg = config['model']

    complex_directory_path = preprocess_cfg['complex_directory']
    interface_directory_path = preprocess_cfg.get('interface_directory')
    output_directory = preprocess_cfg['output_directory']
    num_json_files = preprocess_cfg.get('num_json_files', 5)

    # Inference-only mode: skip interface labels and distance maps when no
    # interface directory is provided / it doesn't exist. The resulting JSON
    # will contain only `combined_chains`, which is all that predict_esm3.py needs.
    inference_only = (
        interface_directory_path is None
        or not os.path.isdir(interface_directory_path)
    )

    pad_length = [model_cfg.get('peptide_len', 50), model_cfg.get('protein_len', 500)]
    length_threshold = [10, 10]
    distance_scale = 10  # Normalization scale for distances

    os.makedirs(output_directory, exist_ok=True)

    print("=" * 60)
    print("GeoPep Preprocessing")
    print("=" * 60)
    print(f"Complex directory: {complex_directory_path}")
    print(f"Interface directory: {interface_directory_path}"
          + ("  [missing -> inference-only mode]" if inference_only else ""))
    print(f"Output directory: {output_directory}")
    print(f"Pad length: {pad_length}")
    if inference_only:
        print("Mode: INFERENCE-ONLY (combined_chains only; "
              "skipping posIdx_binary and distance fields)")
    print("=" * 60)

    # Error log for distance calculation
    error_log_path = os.path.join(output_directory, 'error_log.txt')
    error_log = open(error_log_path, 'w', encoding='utf-8')

    pdb_files = [f for f in os.listdir(complex_directory_path) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files")

    pdb_dict = {}
    numLenProblem = 0

    for idx, pdb_file in enumerate(pdb_files):
        filename = pdb_file.replace('.pdb', '')
        parts = filename.split('_')
        pdb_id = parts[0]
        chains = parts[1:] if len(parts) > 1 else []

        if not chains:
            pdb_parser = PDBParser(QUIET=True)
            try:
                structure = pdb_parser.get_structure('temp', os.path.join(complex_directory_path, pdb_file))
                chains = [chain.id for chain in structure[0]]
            except:
                print(f"Skipping {pdb_file}: cannot determine chains")
                continue

        chains_tuple = tuple(chains)
        key = '_'.join(chains)

        if pdb_id not in pdb_dict:
            pdb_dict[pdb_id] = {
                'combined_chains': {},
                'posIdx_wholeComplex': {},
                'posIdx_interface': {},
                'posIdx_binary': {},
                'distance': {}
            }

        # Extract combined chains sequence
        if key not in pdb_dict[pdb_id]['combined_chains']:
            pdb_file_path = os.path.join(complex_directory_path, pdb_file)
            sequence, lenCheck_flag = extract_sequence_combinedChains(pdb_file_path, chains_tuple, pad_length, length_threshold)
            if lenCheck_flag == False:
                numLenProblem += 1
                continue
            pdb_dict[pdb_id]['combined_chains'][key] = f"{'|'.join(sequence[chain] for chain in chains_tuple if chain in sequence)}"

        if not inference_only:
            # Extract interface position indices
            if key not in pdb_dict[pdb_id]['posIdx_interface']:
                pdb_file_path = os.path.join(interface_directory_path, pdb_file)
                if os.path.exists(pdb_file_path):
                    sequence = extract_sequence_posIdx(pdb_file_path, chains_tuple)
                    pdb_dict[pdb_id]['posIdx_interface'][key] = ' | '.join(sequence[chain] for chain in chains_tuple if chain in sequence)
                else:
                    print(f"Interface file not found: {pdb_file_path}")
                    continue

            # Extract whole complex position indices
            if key not in pdb_dict[pdb_id]['posIdx_wholeComplex']:
                pdb_file_path = os.path.join(complex_directory_path, pdb_file)
                sequence = extract_sequence_posIdx(pdb_file_path, chains_tuple)
                pdb_dict[pdb_id]['posIdx_wholeComplex'][key] = ' | '.join(sequence[chain] for chain in chains_tuple if chain in sequence)

            # Generate binary string
            if key not in pdb_dict[pdb_id]['posIdx_binary']:
                str_wholeComplex = pdb_dict[pdb_id]['posIdx_wholeComplex'][key]
                str_interface = pdb_dict[pdb_id]['posIdx_interface'][key]
                str_binary = generate_binary_string(pdb_id, str_wholeComplex, str_interface, pad_length)
                pdb_dict[pdb_id]['posIdx_binary'][key] = str_binary

            # Calculate distance map
            if key not in pdb_dict[pdb_id]['distance']:
                posIdx_binary_str = pdb_dict[pdb_id]['posIdx_binary'][key]
                distances = calculate_distance_for_entry(
                    pdb_id, key, posIdx_binary_str,
                    complex_directory_path, pad_length, distance_scale, error_log
                )
                if distances is not None:
                    pdb_dict[pdb_id]['distance'][key] = distances

        print(f"Complete {idx+1} out of {len(pdb_files)}")

    error_log.close()

    # Clean up intermediate data
    for pdb_id in list(pdb_dict.keys()):
        if 'posIdx_wholeComplex' in pdb_dict[pdb_id]:
            del pdb_dict[pdb_id]['posIdx_wholeComplex']
        if 'posIdx_interface' in pdb_dict[pdb_id]:
            del pdb_dict[pdb_id]['posIdx_interface']
        if pdb_dict[pdb_id]['combined_chains'] == {}:
            del pdb_dict[pdb_id]

    # Split into multiple JSON files
    pdb_ids = list(pdb_dict.keys())
    chunk_size = max(1, len(pdb_ids) // num_json_files + 1)
    output_files = []

    for i in range(num_json_files):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(pdb_ids))

        chunk_ids = pdb_ids[start:end]
        chunk_data = {pid: pdb_dict[pid] for pid in chunk_ids}

        output_path = os.path.join(output_directory, f"data_part_{i+1}.json")
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(chunk_data, json_file, indent=4)

        output_files.append(output_path)
        print(f"Saved {output_path} ({len(chunk_ids)} entries)")

    print(f"\nPreprocessing complete!")
    print(f"Total entries: {len(pdb_dict)}")
    print(f"Skipped due to length issues: {numLenProblem}")
    print(f"Error log saved to: {error_log_path}")

    return output_files


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDB files to JSON")
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    resolve_paths(config, args.config)

    run_preprocessing(config)


if __name__ == "__main__":
    main()
