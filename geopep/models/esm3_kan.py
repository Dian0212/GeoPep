"""
ESM3-KAN Model for Training

PyTorch Lightning module combining ESM3 backbone with KAN layers.
Supports multi-GPU training with FSDP.
"""

import torch
import torch.nn as nn
import lightning as L
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_model_tokenizers
from fastkan import FastKAN as KAN


class ESM3KAN(L.LightningModule):
    """
    ESM3 with Kolmogorov-Arnold Network heads for binding site prediction.
    
    Args:
        pred_len: Total prediction length (peptide_len + protein_len + 1)
        num_label_types: Number of output classes (default: 3)
        peptide_len: Maximum peptide length (default: 50)
        protein_len: Maximum protein length (default: 500)
        learning_rate: Learning rate (default: 1e-4)
        use_distance_loss: Whether to use distance-based loss (default: True)
    """
    
    def __init__(
        self,
        pred_len: int = 551,
        num_label_types: int = 3,
        peptide_len: int = 50,
        protein_len: int = 500,
        learning_rate: float = 1e-4,
        use_distance_loss: bool = True
    ):
        super(ESM3KAN, self).__init__()
        self.save_hyperparameters()
        
        # Load ESM3 model
        self.esm3_model = ESM3_sm_open_v0()
        self.tokenizers = get_model_tokenizers()
        
        # ESM3 embedding dimension
        esm3_embed_dim = 1536
        
        # KAN layers - 5 sequential modules
        self.KAN_model1 = KAN([esm3_embed_dim, 495, 1486])
        self.KAN_model2 = KAN([1486, 445, 1338])
        self.KAN_model3 = KAN([1338, 396, 1191])
        self.KAN_model4 = KAN([1191, 347, 1043])
        self.KAN_model5 = KAN([1043, 495, num_label_types * pred_len])
        
        self.flatten = nn.Flatten()
        self.predLen = pred_len
        self.numLabelTypes = num_label_types
        self.peptide_len = peptide_len
        self.protein_len = protein_len
        self.learning_rate = learning_rate
        self.use_distance_loss = use_distance_loss
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass through ESM3 and KAN layers."""
        x = self.esm3_model.forward(sequence_tokens=x)
        x = x.embeddings[:, 0, :]  # CLS token: [batch, 1536]
        
        # Pass through KAN layers
        x = self.KAN_model1(x)
        x = self.KAN_model2(x)
        x = self.KAN_model3(x)
        x = self.KAN_model4(x)
        x = self.KAN_model5(x)
        
        out = x.view(-1, self.numLabelTypes, self.predLen)
        return out

    def training_step(self, batch, batch_idx):
        """Training step."""
        if self.use_distance_loss:
            (combined_seq, labels, pep_dist, prot_dist,
             pep_dist_bin, prot_dist_bin, pep_dist_bin2,
             prot_dist_bin2, sample_id) = batch[:9]
        else:
            combined_seq, labels, sample_id = batch[0], batch[1], batch[-1]

        # Tokenize sequences
        tokenized = torch.tensor(
            [self.tokenizers.sequence.encode(seq) for seq in combined_seq],
            dtype=torch.int64, device=self.device
        )

        # Forward pass
        outputs = self(tokenized)

        # Split outputs and labels - SKIP position 50 (separator with label 3)
        # Peptide: positions 0-49, Protein: positions 51-550
        out_peptide = outputs[:, :, :self.peptide_len]  # [batch, num_classes, 50]
        out_protein = outputs[:, :, self.peptide_len + 1:]  # [batch, num_classes, 500]

        labels_peptide = labels[:, :self.peptide_len]  # [batch, 50]
        labels_protein = labels[:, self.peptide_len + 1:]  # [batch, 500]

        # Compute loss separately for peptide and protein (skipping separator)
        loss_peptide = self.criterion(out_peptide, labels_peptide)
        loss_protein = self.criterion(out_protein, labels_protein)
        loss = loss_peptide + loss_protein

        if self.use_distance_loss:
            distance_loss = self._compute_distance_loss(
                outputs, pep_dist, prot_dist, pep_dist_bin, prot_dist_bin
            )
            loss = loss + distance_loss
            self.log('train_distance_loss', distance_loss, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if self.use_distance_loss:
            (combined_seq, labels, pep_dist, prot_dist,
             pep_dist_bin, prot_dist_bin, pep_dist_bin2,
             prot_dist_bin2, sample_id) = batch[:9]
        else:
            combined_seq, labels, sample_id = batch[0], batch[1], batch[-1]

        tokenized = torch.tensor(
            [self.tokenizers.sequence.encode(seq) for seq in combined_seq],
            dtype=torch.int64, device=self.device
        )

        outputs = self(tokenized)

        # Split outputs and labels - SKIP position 50 (separator with label 3)
        out_peptide = outputs[:, :, :self.peptide_len]
        out_protein = outputs[:, :, self.peptide_len + 1:]

        labels_peptide = labels[:, :self.peptide_len]
        labels_protein = labels[:, self.peptide_len + 1:]

        # Compute loss separately for peptide and protein
        loss_peptide = self.criterion(out_peptide, labels_peptide)
        loss_protein = self.criterion(out_protein, labels_protein)
        loss = loss_peptide + loss_protein

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def _compute_distance_loss(self, outputs, pep_dist, prot_dist, pep_dist_bin, prot_dist_bin):
        """Compute distance-based geometric loss."""
        pred = outputs.argmax(dim=1)
        
        # Peptide distance loss
        pep_pred = pred[:, :self.peptide_len]
        pep_loss = ((pep_pred == 1).float() * pep_dist[:, :self.peptide_len]).mean()
        
        # Protein distance loss
        prot_start = self.peptide_len + 1
        prot_pred = pred[:, prot_start:prot_start + self.protein_len]
        prot_loss = ((prot_pred == 1).float() * prot_dist).mean()
        
        return pep_loss + prot_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
