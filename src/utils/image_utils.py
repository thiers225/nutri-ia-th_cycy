"""
Utilitaires image pour le pipeline Nutri-IA.

Fournit :
- Les transformations torchvision pour l'entraînement et l'inférence.
- Un Dataset PyTorch pour charger un répertoire d'images.
- Des helpers de visualisation (grille d'images, voisins par similarité).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms

from src.config.settings import embed_cfg

# Extensions d'images acceptées
VALID_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def get_train_transforms(
    image_size: int | None = None,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> transforms.Compose:
    """
    Pipeline d'augmentation pour l'entraînement.

    Inclut : redimensionnement aléatoire, flip horizontal, légères
    variations de couleur et normalisation ImageNet.

    Args:
        image_size: Taille cible (carré). Par défaut : ``embed_cfg.image_size``.
        mean: Moyenne de normalisation. Par défaut : ``embed_cfg.mean``.
        std: Écart-type de normalisation. Par défaut : ``embed_cfg.std``.

    Returns:
        Composition de transformations torchvision.
    """
    sz = image_size or embed_cfg.image_size
    mu = mean or embed_cfg.mean
    sigma = std or embed_cfg.std

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(sz, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mu), std=list(sigma)),
        ]
    )


def get_inference_transforms(
    image_size: int | None = None,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> transforms.Compose:
    """
    Pipeline déterministe pour l'inférence / génération d'embeddings.

    Args:
        image_size: Taille cible (carré). Par défaut : ``embed_cfg.image_size``.
        mean: Moyenne de normalisation. Par défaut : ``embed_cfg.mean``.
        std: Écart-type de normalisation. Par défaut : ``embed_cfg.std``.

    Returns:
        Composition de transformations torchvision.
    """
    sz = image_size or embed_cfg.image_size
    mu = mean or embed_cfg.mean
    sigma = std or embed_cfg.std

    return transforms.Compose(
        [
            transforms.Resize(int(sz * 1.14)),   # légère marge avant crop centré
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mu), std=list(sigma)),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _collect_image_paths(root: Path, recursive: bool = True) -> list[Path]:
    """Retourne la liste de tous les fichiers image valides sous ``root``."""
    pattern = "**/*" if recursive else "*"
    paths = [
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(paths)


class FoodImageDataset(Dataset):
    """
    Dataset PyTorch pour un répertoire d'images alimentaires.

    Supporte une structure plate ou hiérarchique (sous-dossiers = classes).
    Les images corrompues sont ignorées avec un avertissement.

    Args:
        root_dir: Chemin racine du répertoire d'images.
        transform: Transformations appliquées à chaque image. Si ``None``,
            utilise les transformations d'inférence par défaut.
        recursive: Si ``True``, parcourt les sous-dossiers.
        class_to_idx: Mapping classe → indice. Si ``None``, inféré depuis
            la structure des dossiers (``None`` si structure plate).

    Attributes:
        image_paths (list[Path]): Chemins des images valides.
        labels (list[int | None]): Indices de classe ou ``None`` si inconnu.
        classes (list[str]): Liste des noms de classes (vide si structure plate).
        class_to_idx (dict[str, int]): Mapping classe → indice.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        recursive: bool = True,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Répertoire introuvable : {self.root_dir}")

        self.transform = transform or get_inference_transforms()

        # Détection automatique de la structure
        subdirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if subdirs and class_to_idx is None:
            # Structure hiérarchique : sous-dossiers = classes
            self.classes = sorted(d.name for d in subdirs)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.classes = list(class_to_idx.keys()) if class_to_idx else []
            self.class_to_idx = class_to_idx or {}

        all_paths = _collect_image_paths(self.root_dir, recursive=recursive)
        self.image_paths: list[Path] = []
        self.labels: list[int | None] = []

        for p in all_paths:
            label: int | None = None
            if self.class_to_idx:
                # Le parent immédiat est la classe
                cls = p.parent.name
                label = self.class_to_idx.get(cls)
            self.image_paths.append(p)
            self.labels.append(label)

        logger.info(
            f"FoodImageDataset — {len(self.image_paths)} images trouvées "
            f"({len(self.classes)} classes) dans {self.root_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning(f"Image corrompue ignorée : {path} ({exc})")
            # Retourne une image noire de remplacement pour ne pas casser le batch
            image = Image.new("RGB", (embed_cfg.image_size, embed_cfg.image_size))

        tensor = self.transform(image)
        sample = {"image": tensor, "path": str(path)}
        if label is not None:
            sample["label"] = label
        return sample


# ---------------------------------------------------------------------------
# Helpers de visualisation (optionnels, nécessitent matplotlib)
# ---------------------------------------------------------------------------

def show_image_grid(
    paths: Sequence[str | Path],
    titles: Sequence[str] | None = None,
    n_cols: int = 4,
    figsize_per_img: tuple[float, float] = (3.0, 3.0),
) -> None:
    """
    Affiche une grille d'images PIL.

    Args:
        paths: Chemins vers les images à afficher.
        titles: Titres optionnels pour chaque image.
        n_cols: Nombre de colonnes dans la grille.
        figsize_per_img: Taille (largeur, hauteur) en pouces par image.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib requis pour show_image_grid.")
        return

    n = len(paths)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_img[0] * n_cols, figsize_per_img[1] * n_rows),
    )
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            try:
                img = Image.open(paths[i]).convert("RGB")
                ax.imshow(img)
                ax.set_title(titles[i] if titles else Path(paths[i]).stem, fontsize=8)
            except Exception:
                ax.text(0.5, 0.5, "Erreur", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
