"""
Configuration centrale du projet Nutri-IA.

Toutes les constantes métier, chemins et hyperparamètres du modèle d'embedding
sont regroupés ici via Pydantic Settings pour permettre une surcharge par
variables d'environnement ou fichier .env.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Chemins racine (résolus par rapport à ce fichier)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]


class Paths:
    """Chemins canoniques du projet (lecture seule)."""

    root: Path = _ROOT

    # Données
    data: Path = _ROOT / "data"
    raw_images: Path = _ROOT / "data" / "raw" / "images"
    interim_images: Path = _ROOT / "data" / "interim" / "images"
    processed_images: Path = _ROOT / "data" / "processed" / "images"
    processed_tabular: Path = _ROOT / "data" / "processed" / "tabular"

    # Artefacts du modèle
    models: Path = _ROOT / "models"
    embeddings: Path = _ROOT / "data" / "processed" / "embeddings"

    @classmethod
    def ensure_all(cls) -> None:
        """Crée tous les répertoires manquants."""
        for attr, value in vars(cls).items():
            if isinstance(value, Path) and attr != "root":
                value.mkdir(parents=True, exist_ok=True)


class EmbeddingModelSettings(BaseSettings):
    """Hyperparamètres et options du modèle d'embedding d'images."""

    model_config = SettingsConfigDict(
        env_prefix="NUTRIA_EMBED_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Backbone ---
    backbone: str = Field(
        default="efficientnet_b2",
        description=(
            "Architecture backbone timm. Exemples : 'efficientnet_b2', "
            "'resnet50', 'vit_base_patch16_224', 'convnext_tiny'."
        ),
    )
    pretrained: bool = Field(
        default=True,
        description="Charger les poids pré-entraînés sur ImageNet.",
    )

    # --- Tête de projection ---
    embedding_dim: int = Field(
        default=256,
        ge=32,
        le=2048,
        description="Dimension de l'espace d'embedding final.",
    )
    projection_hidden_dim: int = Field(
        default=512,
        ge=64,
        description="Dimension cachée de la tête de projection MLP.",
    )
    projection_dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Taux de dropout dans la tête de projection.",
    )
    normalize_output: bool = Field(
        default=True,
        description="Normaliser l'embedding final (norme L2 = 1).",
    )

    # --- Prétraitement image ---
    image_size: int = Field(
        default=224,
        description="Taille (px) à laquelle redimensionner les images (carré).",
    )
    mean: tuple[float, float, float] = Field(
        default=(0.485, 0.456, 0.406),
        description="Moyenne de normalisation ImageNet (RGB).",
    )
    std: tuple[float, float, float] = Field(
        default=(0.229, 0.224, 0.225),
        description="Écart-type de normalisation ImageNet (RGB).",
    )

    # --- Inférence / export ---
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Taille de batch pour la génération d'embeddings.",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Nombre de workers DataLoader.",
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description=(
            "Device cible. 'auto' sélectionne CUDA > MPS > CPU automatiquement."
        ),
    )
    output_format: Literal["npy", "hdf5", "parquet"] = Field(
        default="hdf5",
        description="Format de sauvegarde des embeddings générés.",
    )
    checkpoint_name: str = Field(
        default="embedding_model.pt",
        description="Nom du fichier de checkpoint sauvegardé dans models/.",
    )


# Instances globales — importées directement par les autres modules
paths = Paths()
embed_cfg = EmbeddingModelSettings()
