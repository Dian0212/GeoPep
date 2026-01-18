#!/usr/bin/env python3
"""
GeoPep Training Script

Usage: python train.py --config ../configs/config.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import sys
sys.path.insert(0, '..')

from geopep.models import ESM3KAN
from geopep.data import PeptideComplexDataset
from geopep.data.dataset import collate_fn


def main():
    parser = argparse.ArgumentParser(description="Train GeoPep")
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Set HuggingFace token if provided
    if 'huggingface' in config and 'token' in config['huggingface']:
        import os
        os.environ['HF_TOKEN'] = config['huggingface']['token']

    print("=" * 60)
    print("GeoPep Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    hw_cfg = config['hardware']
    
    # Create datasets
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

    # Debug: Check label values
    print("\nChecking label values in first training sample...")
    sample = train_dataset[0]
    labels = sample[1]  # Labels are the second element
    print(f"Label shape: {labels.shape}")
    print(f"Unique labels: {torch.unique(labels)}")
    print(f"Label min: {labels.min()}, max: {labels.max()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    model = ESM3KAN(
        learning_rate=train_cfg['learning_rate'],
        use_distance_loss=train_cfg.get('use_distance_loss', True)
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='../model_weights',
        filename='model-{epoch:02d}',
        save_top_k=1,
        mode='min',
        save_weights_only=True,  # Save only weights to reduce memory usage
        every_n_epochs=1  # Save every epoch
    )
    
    # Trainer
    # Use devices=1 for single GPU to avoid distributed training issues
    trainer = L.Trainer(
        max_epochs=train_cfg['max_epochs'],
        accelerator='gpu',
        devices=1,  # Single GPU
        precision=hw_cfg.get('precision', 16),
        callbacks=[checkpoint_callback]
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
