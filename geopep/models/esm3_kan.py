"""
ESM3-KAN Model for Per-Residue Binding Site Prediction.

Each residue's ESM3 embedding (1536-d) is mapped through 5 KAN layers
to per-position class logits (3 classes: 0=non-interface, 1=interface, 2=padding).
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_model_tokenizers
from fastkan import FastKAN as KAN


def _default_kan_layer_sizes(d_in: int, d_out: int, num_layers: int) -> List[int]:
    """Linearly interpolated layer sizes from d_in down to d_out across num_layers."""
    layer_sizes = [d_in]
    for i in range(1, num_layers):
        intermediate = d_in - ((d_in - d_out) * i) // (num_layers - 1)
        layer_sizes.append(intermediate)
    layer_sizes.append(d_out)
    return layer_sizes


class ESM3KAN(L.LightningModule):
    """ESM3 + per-residue KAN heads for binding site prediction.

    Args:
        pred_len: total prediction length (peptide_len + 1 separator + protein_len).
        num_label_types: number of output classes (default 3 — 0/1/2).
        peptide_len: max peptide length.
        protein_len: max protein length.
        learning_rate: AdamW learning rate.
        use_distance_loss: add differentiable distance-based geometric loss term.
        kan_layer_sizes: optional override of the (num_layers + 1) layer widths.
        num_layers: number of stacked KAN modules (default 5).
        peptide_weight / protein_weight: relative CE-loss weights for the two halves.
        ce_class_weights: per-class CE weights; class 2 (padding) should be 0.
    """

    def __init__(
        self,
        pred_len: int = 551,
        num_label_types: int = 3,
        peptide_len: int = 50,
        protein_len: int = 500,
        learning_rate: float = 1e-4,
        use_distance_loss: bool = True,
        kan_layer_sizes: Optional[List[int]] = None,
        num_layers: int = 5,
        peptide_weight: Optional[float] = None,
        protein_weight: Optional[float] = None,
        ce_class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.esm3_model = ESM3_sm_open_v0()
        self.tokenizers = get_model_tokenizers()

        esm3_embed_dim = 1536
        if kan_layer_sizes is None:
            kan_layer_sizes = _default_kan_layer_sizes(
                esm3_embed_dim, num_label_types, num_layers
            )
        assert len(kan_layer_sizes) == num_layers + 1, (
            f"kan_layer_sizes must have length num_layers+1={num_layers + 1}, "
            f"got {len(kan_layer_sizes)}"
        )

        kans = []
        for i in range(num_layers):
            in_size = kan_layer_sizes[i]
            out_size = kan_layer_sizes[i + 1]
            num_neurons = max(16, int(out_size * 0.3))
            kans.append(KAN([in_size, num_neurons, out_size]))
        self.KAN_model1 = kans[0]
        self.KAN_model2 = kans[1]
        self.KAN_model3 = kans[2]
        self.KAN_model4 = kans[3]
        self.KAN_model5 = kans[4]

        self.predLen = pred_len
        self.numLabelTypes = num_label_types
        self.peptide_len = peptide_len
        self.protein_len = protein_len
        self.learning_rate = learning_rate
        self.use_distance_loss = use_distance_loss

        if peptide_weight is None:
            peptide_weight = 10.0 * peptide_len / (pred_len - 1)
        if protein_weight is None:
            protein_weight = 10.0 * protein_len / (pred_len - 1)
        self.peptide_weight = peptide_weight
        self.protein_weight = protein_weight

        if ce_class_weights is None:
            ce_class_weights = [10.0 * 0.2, 10.0 * 0.8, 0.0]
        self.register_buffer(
            "ce_class_weights",
            torch.tensor(ce_class_weights, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-residue forward pass.

        Args:
            x: token ids, shape [batch, predLen + 2] (BOS + predLen residues + EOS).
        Returns:
            logits: [batch, num_label_types, predLen].
        """
        x = self.esm3_model.forward(sequence_tokens=x)
        x = x.embeddings[:, 1:self.predLen + 1, :]  # [batch, predLen, 1536]

        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size * seq_len, -1)
        x = self.KAN_model1(x)
        x = self.KAN_model2(x)
        x = self.KAN_model3(x)
        x = self.KAN_model4(x)
        x = self.KAN_model5(x)
        x = x.reshape(batch_size, seq_len, self.numLabelTypes)
        logits = x.permute(0, 2, 1)  # [batch, num_classes, predLen]
        return logits

    def _tokenize(self, sequences):
        return torch.tensor(
            [self.tokenizers.sequence.encode(seq) for seq in sequences],
            dtype=torch.int64,
            device=self.device,
        )

    def _split_peptide_protein(self, tensor: torch.Tensor):
        """Split a [..., predLen] tensor into peptide [0:peptide_len] and protein [peptide_len+1:]."""
        peptide = tensor[..., :self.peptide_len]
        protein = tensor[..., self.peptide_len + 1:]
        return peptide, protein

    def _shared_step(self, batch, stage: str):
        if self.use_distance_loss:
            (combined_seq, labels, pep_dist, prot_dist,
             _, _, _, _, _) = batch[:9]
        else:
            combined_seq, labels = batch[0], batch[1]
            pep_dist = prot_dist = None

        tokenized = self._tokenize(combined_seq)
        logits = self(tokenized)  # [batch, 3, predLen]

        out_peptide, out_protein = self._split_peptide_protein(logits)
        labels_peptide, labels_protein = self._split_peptide_protein(labels)

        loss_pep = F.cross_entropy(
            out_peptide, labels_peptide, weight=self.ce_class_weights
        )
        loss_prot = F.cross_entropy(
            out_protein, labels_protein, weight=self.ce_class_weights
        )
        loss_prediction = (
            self.peptide_weight * loss_pep + self.protein_weight * loss_prot
        )
        loss = loss_prediction

        if self.use_distance_loss and pep_dist is not None:
            probs = F.softmax(logits, dim=1)
            interface_probs = probs[:, 1, :]
            pep_probs, prot_probs = self._split_peptide_protein(interface_probs)

            pep_dist = pep_dist.float().to(self.device)
            prot_dist = prot_dist.float().to(self.device)

            pep_valid = (pep_dist >= 0).float().sum() + 1e-8
            prot_valid = (prot_dist >= 0).float().sum() + 1e-8
            pep_dist_clamped = torch.clamp(pep_dist, min=0)
            prot_dist_clamped = torch.clamp(prot_dist, min=0)

            pep_dist_loss = (pep_probs * pep_dist_clamped).sum() / pep_valid
            prot_dist_loss = (prot_probs * prot_dist_clamped).sum() / prot_valid
            loss_distance = pep_dist_loss + prot_dist_loss
            loss = loss + loss_distance

            self.log(f"{stage}_distance_loss", loss_distance,
                     prog_bar=True, sync_dist=True)

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_pred_loss", loss_prediction, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
