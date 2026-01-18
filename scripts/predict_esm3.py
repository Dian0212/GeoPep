#!/usr/bin/env python3
"""
GeoPep Prediction Script

Usage: python predict_esm3.py --config ../configs/config.yaml
"""

import os
import torch
import torch.nn.functional as F
import json
import argparse
import yaml
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_model_tokenizers
from fastkan import FastKAN as KAN
import lightning as L


class esm3_KAN(L.LightningModule):
    """ESM3-KAN model for inference."""
    
    def __init__(self, esm3_model, KAN_models, predLen, numLabelTypes, peptide_len, protein_len):
        super(esm3_KAN, self).__init__()
        self.esm3_model = esm3_model.eval()
        
        self.KAN_model1 = KAN_models[0]
        self.KAN_model2 = KAN_models[1]
        self.KAN_model3 = KAN_models[2]
        self.KAN_model4 = KAN_models[3]
        self.KAN_model5 = KAN_models[4]
        
        self.flatten = torch.nn.Flatten()
        self.predLen = predLen
        self.numLabelTypes = numLabelTypes
        self.peptide_len = peptide_len
        self.protein_len = protein_len

    def forward(self, x):
        x = self.esm3_model.forward(sequence_tokens=x)
        x = x.embeddings[:, 0, :]

        x = self.KAN_model1(x)
        x = self.KAN_model2(x)
        x = self.KAN_model3(x)
        x = self.KAN_model4(x)
        x = self.KAN_model5(x)
        
        out = x.view(-1, self.numLabelTypes, self.predLen)
        return out


def analyze_checkpoint_kan_structure(checkpoint_path):
    """Analyze checkpoint to get KAN layer sizes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    kan_configs = []
    for i in range(1, 6):
        kan_name = f"KAN_model{i}"
        layer0_base = state_dict[f"{kan_name}.layers.0.base_linear.weight"].shape
        layer1_base = state_dict[f"{kan_name}.layers.1.base_linear.weight"].shape
        
        input_size = layer0_base[1]
        hidden_size = layer0_base[0]
        output_size = layer1_base[0]
        
        kan_config = [input_size, hidden_size, output_size]
        kan_configs.append(kan_config)
        print(f"{kan_name} config: {kan_config}")
    
    return kan_configs


def load_model(checkpoint_path, device="cpu"):
    """Load model with correct architecture from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    
    state_dict_filtered = {k: v for k, v in state_dict.items() if not k.startswith("criterion")}
    print("Removed criterion parameters")
    
    print("Analyzing checkpoint KAN structure...")
    kan_configs = analyze_checkpoint_kan_structure(checkpoint_path)
    
    print("Loading ESM3 model...")
    tokenizers = get_model_tokenizers()
    esm3_model = ESM3_sm_open_v0()
    
    print("Creating KAN models...")
    KAN_models = []
    for i, config in enumerate(kan_configs):
        print(f"Creating KAN_model{i+1} with config: {config}")
        KAN_models.append(KAN(config))
    
    numLabelTypes = 3
    protein_len = 500
    peptide_len = 50
    predLen = protein_len + peptide_len + 1
    
    model = esm3_KAN(esm3_model, KAN_models, predLen, numLabelTypes, peptide_len, protein_len)
    
    try:
        model.load_state_dict(state_dict_filtered, strict=True)
        print("✅ Model loaded successfully!")
    except RuntimeError as e:
        print(f"❌ Failed to load: {e}")
        return None, None
    
    model = model.to(device)
    model.eval()
    return model, tokenizers


def run_inference(json_file_path, model, tokenizers, device="cpu", input_field="model_in"):
    """Run inference on all samples."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    total_pdb_ids = len(json_data)
    print(f"Starting inference on {total_pdb_ids} PDB entries...")

    sample_count = 0
    for idx, (pdb_id, data) in enumerate(json_data.items(), 1):
        print(f"\n{'='*60}")
        print(f"Processing PDB {idx}/{total_pdb_ids} - {pdb_id}")
        print(f"{'='*60}")

        # Handle nested structure: combined_chains is a dict with chain keys
        if "combined_chains" in data and isinstance(data["combined_chains"], dict):
            # Initialize output dict if not exists
            if "model_out_argmax" not in data:
                data["model_out_argmax"] = {}
            if "model_out_softmax" not in data:
                data["model_out_softmax"] = {}

            for chain_key, model_in in data["combined_chains"].items():
                sample_count += 1
                print(f"\n  Chain: {chain_key}")
                print(f"  Input sequence length: {len(model_in)}")

                tokenized_seq = torch.tensor(
                    [tokenizers.sequence.encode(model_in)],
                    dtype=torch.int64
                ).to(device)
                print(f"  Tokenized sequence shape: {tokenized_seq.shape}")

                with torch.no_grad():
                    output = model(tokenized_seq)
                    print(f"  Model output shape: {output.shape}")

                    predicted_labels = output.argmax(dim=1).cpu()
                    predicted_labels_list = predicted_labels.squeeze(0).tolist()
                    argmax_string = ''.join([str(int(x)) for x in predicted_labels_list])

                    print(f"  Argmax string length: {len(argmax_string)}")

                    unique_values, counts = torch.unique(predicted_labels.squeeze(0), return_counts=True)
                    print(f"  Class distribution:")
                    for val, count in zip(unique_values, counts):
                        print(f"    Class {int(val)}: {int(count)} positions ({int(count)/551*100:.1f}%)")

                softmax_output = F.softmax(output, dim=1)
                softmax_probs = softmax_output.cpu().tolist()

                # Save predictions with same chain key structure
                data["model_out_argmax"][chain_key] = predicted_labels_list
                data["model_out_softmax"][chain_key] = softmax_probs

                print(f"  ✅ Completed processing for {pdb_id}_{chain_key}")
        else:
            # Handle flat structure (single sequence per entry)
            model_in = data.get(input_field) or data.get("model_in") or data.get("combined_chains")
            if model_in is None:
                print(f"Warning: No input field found for {pdb_id}, skipping...")
                continue

            sample_count += 1
            print(f"Input sequence length: {len(model_in)}")

            tokenized_seq = torch.tensor(
                [tokenizers.sequence.encode(model_in)],
                dtype=torch.int64
            ).to(device)

            with torch.no_grad():
                output = model(tokenized_seq)
                predicted_labels = output.argmax(dim=1).cpu()
                predicted_labels_list = predicted_labels.squeeze(0).tolist()

            softmax_output = F.softmax(output, dim=1)
            softmax_probs = softmax_output.cpu().tolist()

            data["model_out_argmax"] = predicted_labels_list
            data["model_out_softmax"] = softmax_probs

            print(f"✅ Completed processing for {pdb_id}")

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print(f"\n{'='*80}")
    print(f"✅ ALL {sample_count} samples from {total_pdb_ids} PDB entries saved in: {json_file_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="GeoPep Prediction")
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Set HuggingFace token if provided
    if 'huggingface' in config and 'token' in config['huggingface']:
        import os
        os.environ['HF_TOKEN'] = config['huggingface']['token']

    pred_cfg = config['prediction']
    
    checkpoint_path = pred_cfg['checkpoint_path']
    input_json = pred_cfg['input_json']
    device = pred_cfg.get('device', 'cpu')
    input_field = pred_cfg.get('input_field', 'model_in')
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print("=" * 80)
    print("GeoPep Prediction")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input JSON: {input_json}")
    print(f"Device: {device}")
    print("=" * 80)
    
    model, tokenizers = load_model(checkpoint_path, device)
    if model is None:
        print("❌ Failed to load model.")
        return
    
    print("\n" + "=" * 80)
    print("STARTING INFERENCE")
    print("=" * 80)
    
    run_inference(input_json, model, tokenizers, device, input_field)


if __name__ == "__main__":
    main()
