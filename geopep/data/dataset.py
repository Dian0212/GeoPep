"""
Dataset classes for peptide-protein binding site prediction.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class PeptideComplexDataset(Dataset):
    """
    Dataset for peptide-protein complexes.
    
    Loads data from JSON files containing:
    - combined_chains: Peptide|Protein sequences
    - posIdx_binary: Binary binding site labels
    - distance: Distance to nearest interface residue
    - distances_binary: Binary distance labels
    
    Args:
        json_paths: List of paths to JSON data files
        peptide_len: Maximum peptide length (default: 50)
        protein_len: Maximum protein length (default: 500)
        return_distances: Whether to return distance information
    """
    
    def __init__(
        self,
        json_paths: List[str],
        peptide_len: int = 50,
        protein_len: int = 500,
        return_distances: bool = True
    ):
        self.peptide_len = peptide_len
        self.protein_len = protein_len
        self.pred_len = peptide_len + protein_len + 1  # +1 for separator
        self.return_distances = return_distances
        
        # Load all data
        self.data = []
        self.keys = []
        
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            for pdb_id, pdb_data in json_data.items():
                for chain_key in pdb_data.get('combined_chains', {}).keys():
                    combined_seq = pdb_data['combined_chains'].get(chain_key)
                    binary_str = pdb_data.get('posIdx_binary', {}).get(chain_key)
                    distances = pdb_data.get('distance', {}).get(chain_key)
                    dist_binary = pdb_data.get('distances_binary', {}).get(chain_key)
                    dist_binary2 = pdb_data.get('distances_binary2', {}).get(chain_key)
                    
                    if combined_seq and binary_str:
                        self.data.append({
                            'combined_seq': combined_seq,
                            'binary_str': binary_str,
                            'distances': distances,
                            'dist_binary': dist_binary,
                            'dist_binary2': dist_binary2,
                            'pdb_id': pdb_id,
                            'chain_key': chain_key
                        })
                        self.keys.append(f"{pdb_id}_{chain_key}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        item = self.data[idx]
        
        # Combined sequence (peptide|protein)
        combined_seq = item['combined_seq']
        
        # Parse binary labels
        binary_parts = item['binary_str'].split()
        labels = torch.tensor([int(x) for x in binary_parts], dtype=torch.long)
        
        # Sample ID
        sample_id = self.keys[idx]
        
        if not self.return_distances:
            return combined_seq, labels, sample_id
        
        # Parse distances
        distances = item['distances'] if item['distances'] else [0] * self.pred_len
        distances = torch.tensor(distances, dtype=torch.float32)
        
        # Split distances for peptide and protein
        peptide_dist = distances[:self.peptide_len]
        protein_dist = distances[self.peptide_len + 1:self.peptide_len + 1 + self.protein_len]
        
        # Parse binary distances
        if item['dist_binary']:
            dist_binary = torch.tensor(item['dist_binary'], dtype=torch.long)
        else:
            dist_binary = labels.clone()
        
        if item['dist_binary2']:
            dist_binary2 = torch.tensor(item['dist_binary2'], dtype=torch.long)
        else:
            dist_binary2 = labels.clone()
        
        # Split binary distances
        pep_dist_bin = dist_binary[:self.peptide_len]
        prot_dist_bin = dist_binary[self.peptide_len + 1:self.peptide_len + 1 + self.protein_len]
        pep_dist_bin2 = dist_binary2[:self.peptide_len]
        prot_dist_bin2 = dist_binary2[self.peptide_len + 1:self.peptide_len + 1 + self.protein_len]
        
        return (
            combined_seq,
            labels,
            peptide_dist,
            protein_dist,
            pep_dist_bin,
            prot_dist_bin,
            pep_dist_bin2,
            prot_dist_bin2,
            sample_id
        )


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    if len(batch[0]) == 3:
        # Without distances
        seqs, labels, ids = zip(*batch)
        labels = torch.stack(labels)
        return list(seqs), labels, list(ids)
    else:
        # With distances
        (seqs, labels, pep_dist, prot_dist, 
         pep_bin, prot_bin, pep_bin2, prot_bin2, ids) = zip(*batch)
        
        labels = torch.stack(labels)
        pep_dist = torch.stack(pep_dist)
        prot_dist = torch.stack(prot_dist)
        pep_bin = torch.stack(pep_bin)
        prot_bin = torch.stack(prot_bin)
        pep_bin2 = torch.stack(pep_bin2)
        prot_bin2 = torch.stack(prot_bin2)
        
        return (list(seqs), labels, pep_dist, prot_dist,
                pep_bin, prot_bin, pep_bin2, prot_bin2, list(ids))
