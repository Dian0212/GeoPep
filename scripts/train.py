#!/usr/bin/env python3
"""
GeoPep Training Script

Usage: python train.py --config ../configs/config.yaml
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.insert(0, '..')

from geopep.models import ESM3KAN
from geopep.data import PeptideComplexDataset
from geopep.data.dataset import collate_fn
from geopep.hf_auth import resolve_hf_token
from geopep.config_utils import resolve_paths


def run_training(config):
    """Run training given a parsed config dict.

    Reads `data`, `model`, `training`, `hardware`, `huggingface` sections.
    Saves checkpoints to `training.checkpoint_dir` (default '../model_weights').
    Returns the path to the best checkpoint.
    """
    resolve_hf_token(config)

    print("=" * 60)
    print("GeoPep Training")
    print("=" * 60)

    data_cfg = config['data']
    train_cfg = config['training']
    hw_cfg = config['hardware']

    checkpoint_dir = train_cfg.get('checkpoint_dir', '../model_weights')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint dir: {checkpoint_dir}")

    train_dataset = PeptideComplexDataset(
        json_paths=data_cfg['train_json'],
        return_distances=train_cfg.get('use_distance_loss', True)
    )
    val_dataset = PeptideComplexDataset(
        json_paths=data_cfg['val_json'],
        return_distances=train_cfg.get('use_distance_loss', True)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Debug: peek at label values in first training sample
    sample = train_dataset[0]
    labels = sample[1]
    print(f"First sample label shape: {labels.shape}, unique: {torch.unique(labels).tolist()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=train_cfg.get('num_workers', 4),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=train_cfg.get('num_workers', 4),
    )

    model = ESM3KAN(
        learning_rate=train_cfg['learning_rate'],
        use_distance_loss=train_cfg.get('use_distance_loss', True),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=train_cfg.get('checkpoint_filename', 'model-{epoch:02d}'),
        save_top_k=train_cfg.get('save_top_k', 1),
        mode='min',
        save_weights_only=True,
        every_n_epochs=1,
    )

    trainer = L.Trainer(
        max_epochs=train_cfg['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=hw_cfg.get('gpus', 1) if torch.cuda.is_available() else 1,
        precision=hw_cfg.get('precision', 16),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    print(f"Training complete. Best checkpoint: {best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train GeoPep")
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    resolve_paths(config, args.config)

    print(f"Config: {args.config}")
    run_training(config)


if __name__ == "__main__":
    main()
