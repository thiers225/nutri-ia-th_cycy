"""
Script de génération d'embeddings sur un dataset d'images.

Usage CLI
---------
    # Générer les embeddings des images processed
    python -m src.process.embed_images

    # Sur un dossier personnalisé, avec un checkpoint existant
    python -m src.process.embed_images \\
        --input  data/processed/images \\
        --output data/processed/embeddings \\
        --checkpoint models/embedding_model.pt \\
        --format hdf5 \\
        --batch-size 32

Sorties
-------
Selon le format choisi :
- ``hdf5``    : un seul fichier ``embeddings.h5``  avec datasets ``embeddings``,
               ``paths`` et éventuellement ``labels``.
- ``npy``     : deux fichiers ``embeddings.npy`` et ``paths.npy``.
- ``parquet`` : un fichier ``embeddings.parquet`` (colonnes e_0..e_N + path).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.settings import embed_cfg, paths
from src.process.embedding_model import FoodEmbeddingModel, resolve_device
from src.utils.image_utils import FoodImageDataset, get_inference_transforms


# ---------------------------------------------------------------------------
# Générateur d'embeddings
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """
    Orchestre la génération d'embeddings sur un dataset d'images.

    Args:
        model: Instance ``FoodEmbeddingModel`` à utiliser.
        device: Device PyTorch (cpu, cuda, mps).
        batch_size: Taille de batch pour le DataLoader.
        num_workers: Nombre de workers DataLoader.
    """

    def __init__(
        self,
        model: FoodEmbeddingModel,
        device: torch.device | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        self.device = device or resolve_device(embed_cfg.device)
        self.batch_size = batch_size or embed_cfg.batch_size
        self.num_workers = num_workers or embed_cfg.num_workers

        self.model = model.to(self.device)
        self.model.eval()
        logger.info(f"EmbeddingGenerator prêt sur {self.device}.")

    def generate(
        self,
        dataset: FoodImageDataset,
    ) -> dict[str, np.ndarray]:
        """
        Génère les embeddings pour toutes les images du dataset.

        Args:
            dataset: Instance ``FoodImageDataset``.

        Returns:
            Dictionnaire avec :
            - ``"embeddings"``  : ``np.ndarray`` shape ``(N, D)``
            - ``"paths"``       : ``np.ndarray`` de strings, shape ``(N,)``
            - ``"labels"``      : ``np.ndarray`` int ou ``None``
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        all_embeddings: list[np.ndarray] = []
        all_paths: list[str] = []
        all_labels: list[int] = []
        has_labels = False

        start = time.perf_counter()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Génération embeddings", unit="batch"):
                images = batch["image"].to(self.device, non_blocking=True)
                embeddings = self.model.encode(images, return_numpy=True)

                all_embeddings.append(embeddings)
                all_paths.extend(batch["path"])

                if "label" in batch:
                    has_labels = True
                    all_labels.extend(batch["label"].tolist())

        elapsed = time.perf_counter() - start
        n = sum(e.shape[0] for e in all_embeddings)
        logger.info(
            f"{n} embeddings générés en {elapsed:.1f}s "
            f"({n / elapsed:.0f} img/s) — dim={all_embeddings[0].shape[1]}"
        )

        result: dict[str, np.ndarray] = {
            "embeddings": np.vstack(all_embeddings).astype(np.float32),
            "paths": np.array(all_paths, dtype=object),
        }
        if has_labels:
            result["labels"] = np.array(all_labels, dtype=np.int32)

        return result

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        data: dict[str, np.ndarray],
        output_dir: str | Path,
        fmt: str | None = None,
    ) -> Path:
        """
        Persiste les embeddings dans le format désiré.

        Args:
            data: Dictionnaire retourné par ``generate()``.
            output_dir: Répertoire de sortie.
            fmt: ``'hdf5'``, ``'npy'`` ou ``'parquet'``.
                 Par défaut : ``embed_cfg.output_format``.

        Returns:
            Chemin du fichier principal créé.

        Raises:
            ValueError: Format inconnu.
        """
        fmt = fmt or embed_cfg.output_format
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "hdf5":
            return EmbeddingGenerator._save_hdf5(data, output_dir)
        elif fmt == "npy":
            return EmbeddingGenerator._save_npy(data, output_dir)
        elif fmt == "parquet":
            return EmbeddingGenerator._save_parquet(data, output_dir)
        else:
            raise ValueError(f"Format inconnu : '{fmt}'. Valeurs acceptées : hdf5, npy, parquet.")

    @staticmethod
    def _save_hdf5(data: dict[str, np.ndarray], output_dir: Path) -> Path:
        out = output_dir / "embeddings.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset(
                "embeddings",
                data=data["embeddings"],
                compression="gzip",
                compression_opts=4,
            )
            paths_encoded = np.array(
                [str(p).encode("utf-8") for p in data["paths"]], dtype=object
            )
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("paths", data=paths_encoded, dtype=dt)
            if "labels" in data:
                f.create_dataset("labels", data=data["labels"])
            # Méta-données
            f.attrs["embedding_dim"] = data["embeddings"].shape[1]
            f.attrs["n_samples"] = data["embeddings"].shape[0]
        logger.info(f"Embeddings sauvegardés (HDF5) → {out}")
        return out

    @staticmethod
    def _save_npy(data: dict[str, np.ndarray], output_dir: Path) -> Path:
        out_emb = output_dir / "embeddings.npy"
        out_paths = output_dir / "paths.npy"
        np.save(out_emb, data["embeddings"])
        np.save(out_paths, data["paths"])
        if "labels" in data:
            np.save(output_dir / "labels.npy", data["labels"])
        logger.info(f"Embeddings sauvegardés (npy) → {out_emb}")
        return out_emb

    @staticmethod
    def _save_parquet(data: dict[str, np.ndarray], output_dir: Path) -> Path:
        out = output_dir / "embeddings.parquet"
        dim = data["embeddings"].shape[1]
        cols = {f"e_{i}": data["embeddings"][:, i] for i in range(dim)}
        cols["path"] = data["paths"].tolist()
        if "labels" in data:
            cols["label"] = data["labels"].tolist()
        df = pd.DataFrame(cols)
        df.to_parquet(out, index=False)
        logger.info(f"Embeddings sauvegardés (parquet) → {out}")
        return out


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def run_embedding_pipeline(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_format: str | None = None,
    batch_size: int | None = None,
    freeze_backbone: bool = True,
) -> Path:
    """
    Pipeline complet : chargement modèle → dataset → génération → sauvegarde.

    Args:
        input_dir: Répertoire d'images source. Défaut : ``data/processed/images``.
        output_dir: Répertoire de sortie. Défaut : ``data/processed/embeddings``.
        checkpoint: Checkpoint ``.pt``. Si ``None``, crée un nouveau modèle.
        output_format: Format de sortie (hdf5 / npy / parquet).
        batch_size: Taille de batch.
        freeze_backbone: Geler le backbone (mode extraction pure).

    Returns:
        Chemin du fichier d'embeddings produit.
    """
    input_dir = Path(input_dir or paths.processed_images)
    output_dir = Path(output_dir or paths.embeddings)

    # Chargement du modèle
    if checkpoint and Path(checkpoint).exists():
        model = FoodEmbeddingModel.load(checkpoint)
    else:
        logger.info("Aucun checkpoint fourni — initialisation depuis la config.")
        model = FoodEmbeddingModel.from_config(freeze_backbone=freeze_backbone)

    # Dataset
    transform = get_inference_transforms()
    dataset = FoodImageDataset(root_dir=input_dir, transform=transform)

    if len(dataset) == 0:
        logger.warning(f"Aucune image trouvée dans {input_dir}. Pipeline arrêté.")
        return output_dir

    # Génération
    device = resolve_device(embed_cfg.device)
    generator = EmbeddingGenerator(model=model, device=device, batch_size=batch_size)
    data = generator.generate(dataset)

    # Sauvegarde
    out_path = EmbeddingGenerator.save(data, output_dir, fmt=output_format)
    return out_path


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Génère les embeddings d'un dataset d'images alimentaires.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Répertoire source des images (processed/images par défaut).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Répertoire de sortie des embeddings.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Chemin vers un checkpoint .pt existant.",
    )
    parser.add_argument(
        "--format", dest="output_format",
        choices=["hdf5", "npy", "parquet"], default=None,
        help="Format de sauvegarde des embeddings.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Taille de batch pour l'inférence.",
    )
    parser.add_argument(
        "--no-freeze", action="store_true",
        help="Ne pas geler le backbone (utile en fine-tuning).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    run_embedding_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        output_format=args.output_format,
        batch_size=args.batch_size,
        freeze_backbone=not args.no_freeze,
    )
