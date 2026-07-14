"""
Modèle d'embedding d'images pour Nutri-IA.

Architecture
------------
┌─────────────────────────────────────────────────────────────────────┐
│  Image RGB (B × 3 × H × W)                                         │
│       │                                                             │
│  ┌────▼──────────────────┐                                          │
│  │  Backbone (timm)      │  ex. EfficientNet-B2, ResNet-50, ViT     │
│  │  Frozen ou Fine-tuné  │                                          │
│  └────┬──────────────────┘                                          │
│       │  feature vector  (B × D_backbone)                          │
│  ┌────▼──────────────────────────────────┐                          │
│  │  Projection Head (MLP)               │                          │
│  │  Linear → BN → ReLU → Dropout        │                          │
│  │  → Linear → BN → ReLU → Dropout      │                          │
│  │  → Linear                             │                          │
│  └────┬──────────────────────────────────┘                          │
│       │  embedding  (B × embedding_dim)                            │
│  ┌────▼──────────────────┐                                          │
│  │  L2 Normalize (opt.)  │                                          │
│  └───────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘

Utilisation typique
-------------------
>>> from src.process.embedding_model import FoodEmbeddingModel
>>> model = FoodEmbeddingModel.from_config()
>>> embeddings = model.encode(images_tensor)   # shape (B, 256)

Le modèle expose trois modes d'utilisation :
1. ``forward()``  — passe complète backbone + projection (entraînement).
2. ``encode()``   — inférence sans gradient, retourne numpy ou tensor.
3. ``extract_features()`` — features brutes du backbone (avant projection).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.config.settings import EmbeddingModelSettings, embed_cfg, paths


# ---------------------------------------------------------------------------
# Résolution du device
# ---------------------------------------------------------------------------

def resolve_device(preference: str = "auto") -> torch.device:
    """
    Résout le device PyTorch en fonction de la préférence et des disponibilités.

    Args:
        preference: ``'auto'``, ``'cuda'``, ``'mps'`` ou ``'cpu'``.

    Returns:
        torch.device sélectionné.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Tête de projection
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    MLP de projection : réduit les features backbone vers l'espace d'embedding.

    Structure : Linear → BN → ReLU → Dropout → Linear → BN → ReLU
                → Dropout → Linear(embedding_dim)

    Args:
        in_features: Dimension d'entrée (sortie du backbone).
        hidden_dim: Dimension des couches intermédiaires.
        out_dim: Dimension de l'embedding de sortie.
        dropout: Taux de dropout appliqué après chaque activation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Couche 1
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Couche 2
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # Couche de sortie (sans activation — appliquée après si besoin)
            nn.Linear(hidden_dim // 2, out_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialisation He pour les couches linéaires."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ---------------------------------------------------------------------------
# Modèle principal
# ---------------------------------------------------------------------------

class FoodEmbeddingModel(nn.Module):
    """
    Modèle d'embedding d'images alimentaires.

    Combine un backbone pré-entraîné (via timm) et une tête de projection MLP
    pour produire des vecteurs d'embedding denses et normalisés.

    Args:
        backbone_name: Identifiant timm du backbone.
        pretrained: Charger les poids ImageNet.
        embedding_dim: Dimension de l'espace d'embedding final.
        projection_hidden_dim: Dimension cachée de la tête de projection.
        projection_dropout: Taux de dropout dans la projection.
        normalize_output: Si ``True``, normalise l'embedding (||e||₂ = 1).
        freeze_backbone: Si ``True``, gèle tous les paramètres du backbone.
        unfreeze_last_n_blocks: Nombre de blocs finaux à dégeler (fine-tuning
            partiel). Ignoré si ``freeze_backbone=False``.

    Raises:
        ValueError: Si ``backbone_name`` n'est pas reconnu par timm.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        projection_dropout: float = 0.2,
        normalize_output: bool = True,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = 0,
    ) -> None:
        super().__init__()

        # --- Backbone ---
        if backbone_name not in timm.list_models():
            raise ValueError(
                f"Backbone '{backbone_name}' non reconnu par timm. "
                f"Consultez timm.list_models() pour la liste complète."
            )
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,   # supprime la tête de classification
            global_pool="avg",
        )
        backbone_out_dim: int = self.backbone.num_features
        logger.info(
            f"Backbone '{backbone_name}' chargé — "
            f"features: {backbone_out_dim}d, pretrained={pretrained}"
        )

        # --- Gel / dégel du backbone ---
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n_blocks)

        # --- Tête de projection ---
        self.projection = ProjectionHead(
            in_features=backbone_out_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=embedding_dim,
            dropout=projection_dropout,
        )

        self.normalize_output = normalize_output
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone_name

    # ------------------------------------------------------------------
    # Gestion du gel des paramètres
    # ------------------------------------------------------------------

    def _freeze_backbone(self, unfreeze_last_n: int = 0) -> None:
        """
        Gèle tous les paramètres du backbone, puis dégèle les ``unfreeze_last_n``
        derniers blocs pour le fine-tuning partiel.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            # timm expose get_classifier() mais pas de liste de blocs unifiée ;
            # on parcourt les enfants directs en sens inverse.
            children = list(self.backbone.children())
            for child in children[-unfreeze_last_n:]:
                for param in child.parameters():
                    param.requires_grad = True
            logger.info(
                f"Backbone gelé — {unfreeze_last_n} dernier(s) bloc(s) dégelé(s)."
            )
        else:
            logger.info("Backbone entièrement gelé (extraction de features uniquement).")

    def unfreeze_backbone(self) -> None:
        """Dégèle tous les paramètres du backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone entièrement dégelé.")

    # ------------------------------------------------------------------
    # Passes forward
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les features brutes du backbone (avant projection).

        Args:
            x: Batch d'images, shape ``(B, 3, H, W)``.

        Returns:
            Features de shape ``(B, D_backbone)``.
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe complète : backbone → projection → normalisation optionnelle.

        Args:
            x: Batch d'images, shape ``(B, 3, H, W)``.

        Returns:
            Embeddings de shape ``(B, embedding_dim)``.
        """
        features = self.backbone(x)          # (B, D_backbone)
        embeddings = self.projection(features)  # (B, embedding_dim)

        if self.normalize_output:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        return_numpy: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """
        Génère les embeddings en mode inférence (pas de gradient).

        Args:
            x: Batch d'images, shape ``(B, 3, H, W)``.
            return_numpy: Si ``True``, retourne un ``np.ndarray`` sur CPU.

        Returns:
            Embeddings de shape ``(B, embedding_dim)``.
        """
        self.eval()
        embeddings = self.forward(x)
        if return_numpy:
            return embeddings.cpu().float().numpy()
        return embeddings

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """
        Sauvegarde les poids du modèle au format ``.pt``.

        Args:
            path: Chemin de sauvegarde. Par défaut : ``models/<checkpoint_name>``.

        Returns:
            Chemin effectif du fichier sauvegardé.
        """
        if path is None:
            paths.models.mkdir(parents=True, exist_ok=True)
            path = paths.models / embed_cfg.checkpoint_name
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "backbone_name": self.backbone_name,
            "embedding_dim": self.embedding_dim,
            "normalize_output": self.normalize_output,
        }
        torch.save(checkpoint, path)
        logger.info(f"Modèle sauvegardé → {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FoodEmbeddingModel":
        """
        Charge un modèle depuis un checkpoint ``.pt``.

        Args:
            path: Chemin vers le fichier de checkpoint.

        Returns:
            Instance ``FoodEmbeddingModel`` prête à l'emploi.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            backbone_name=checkpoint["backbone_name"],
            pretrained=False,   # poids chargés depuis le checkpoint
            embedding_dim=checkpoint["embedding_dim"],
            normalize_output=checkpoint["normalize_output"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Modèle chargé depuis {path}")
        return model

    @classmethod
    def from_config(
        cls,
        cfg: EmbeddingModelSettings | None = None,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = 0,
    ) -> "FoodEmbeddingModel":
        """
        Instancie le modèle à partir de la configuration globale (ou custom).

        Args:
            cfg: Configuration ``EmbeddingModelSettings``. Si ``None``,
                 utilise ``embed_cfg`` (la config globale du projet).
            freeze_backbone: Geler le backbone.
            unfreeze_last_n_blocks: Nombre de blocs finaux à dégeler.

        Returns:
            Instance ``FoodEmbeddingModel`` configurée.
        """
        cfg = cfg or embed_cfg
        return cls(
            backbone_name=cfg.backbone,
            pretrained=cfg.pretrained,
            embedding_dim=cfg.embedding_dim,
            projection_hidden_dim=cfg.projection_hidden_dim,
            projection_dropout=cfg.projection_dropout,
            normalize_output=cfg.normalize_output,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    # ------------------------------------------------------------------
    # Informations
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """
        Compte les paramètres entraînables et total du modèle.

        Returns:
            Dictionnaire ``{'trainable': ..., 'total': ..., 'frozen': ...}``.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "total": total, "frozen": total - trainable}

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"FoodEmbeddingModel(\n"
            f"  backbone={self.backbone_name},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  normalize={self.normalize_output},\n"
            f"  params_total={counts['total']:,},\n"
            f"  params_trainable={counts['trainable']:,}\n"
            f")"
        )
