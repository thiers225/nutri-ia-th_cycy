"""
Script de prétraitement des images brutes.

Copie les images de ``data/raw/images/`` vers ``data/processed/images/``
en appliquant les nettoyages suivants :
- Vérification que l'image est lisible (non corrompue)
- Conversion en RGB (supprime les canaux alpha, gère les images en niveaux de gris)
- Suppression des fichiers système parasites (.DS_Store, Thumbs.db, etc.)
- Conservation de la structure de classes (sous-dossiers = classes)

Usage CLI
---------
    python -m src.process.prepare_images

    python -m src.process.prepare_images \\
        --input  data/raw/images \\
        --output data/processed/images \\
        --max-size 1024 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.config.settings import paths

# Fichiers à ignorer lors du scan
IGNORED_FILES: frozenset[str] = frozenset(
    {".DS_Store", "Thumbs.db", ".gitkeep", "desktop.ini", ".gitignore"}
)

# Extensions d'images acceptées
VALID_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_image_file(path: Path) -> bool:
    """Retourne True si le fichier est une image avec une extension valide."""
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and path.name not in IGNORED_FILES


def _scan_raw_images(raw_dir: Path) -> list[Path]:
    """Liste tous les fichiers image valides sous raw_dir (récursif)."""
    files = [p for p in raw_dir.rglob("*") if _is_valid_image_file(p)]
    return sorted(files)


# ---------------------------------------------------------------------------
# Prétraitement d'une image
# ---------------------------------------------------------------------------

def process_image(
    src: Path,
    dst: Path,
    max_size: int | None = None,
) -> bool:
    """
    Traite une image source et la sauvegarde à destination.

    Opérations appliquées :
    1. Ouverture et vérification (lève une exception si corrompue)
    2. Conversion en RGB (supprime canal alpha, gère grayscale)
    3. Redimensionnement si la plus grande dimension dépasse ``max_size``
    4. Sauvegarde en JPEG qualité 95

    Args:
        src: Chemin de l'image source (raw).
        dst: Chemin de destination (processed).
        max_size: Taille maximale en pixels (hauteur ou largeur). ``None`` = pas de redim.

    Returns:
        ``True`` si le traitement a réussi, ``False`` sinon.
    """
    try:
        img = Image.open(src)

        # Conversion RGB obligatoire : supprime canal alpha (RGBA, LA)
        # et convertit les images en niveaux de gris (L, LA)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Redimensionnement optionnel — préserve le ratio
        if max_size is not None:
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_w = max(1, int(w * ratio))
                new_h = max(1, int(h * ratio))
                img = img.resize((new_w, new_h), Image.LANCZOS)

        # Crée les dossiers parents si nécessaire
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarde en JPEG (même les .png pour uniformiser)
        dst_jpg = dst.with_suffix(".jpg")
        img.save(dst_jpg, format="JPEG", quality=95, optimize=True)
        return True

    except (UnidentifiedImageError, OSError, Exception) as exc:
        logger.warning(f"Image ignorée (erreur) : {src} — {exc}")
        return False


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_prepare_pipeline(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_size: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Pipeline complet de prétraitement : raw → processed.

    Args:
        input_dir:  Répertoire source (raw/images). Défaut : ``paths.raw_images``.
        output_dir: Répertoire de sortie (processed/images). Défaut : ``paths.processed_images``.
        max_size:   Taille max en pixels (plus grande dimension). ``None`` = pas de redim.
        dry_run:    Si ``True``, simule sans écrire de fichiers.
        overwrite:  Si ``False``, ignore les images déjà présentes dans processed.

    Returns:
        Dictionnaire avec les compteurs : ``{'total', 'processed', 'skipped', 'errors'}``.
    """
    input_dir = Path(input_dir or paths.raw_images)
    output_dir = Path(output_dir or paths.processed_images)

    if not input_dir.exists():
        raise FileNotFoundError(f"Répertoire source introuvable : {input_dir}")

    logger.info(f"Source      : {input_dir}")
    logger.info(f"Destination : {output_dir}")
    logger.info(f"Max size    : {max_size or 'aucune limite'}")
    logger.info(f"Dry run     : {dry_run}")
    logger.info(f"Overwrite   : {overwrite}")

    # Scan des images sources
    all_images = _scan_raw_images(input_dir)
    total = len(all_images)
    logger.info(f"{total} image(s) trouvée(s) dans {input_dir}")

    if total == 0:
        logger.warning("Aucune image trouvée. Vérifiez le chemin source.")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}

    # Affichage des classes détectées
    classes = sorted({p.parent.name for p in all_images if p.parent != input_dir})
    if classes:
        logger.info(f"Classes détectées ({len(classes)}) : {', '.join(classes)}")
    else:
        logger.info("Structure plate détectée (pas de sous-dossiers de classes).")

    counters = {"total": total, "processed": 0, "skipped": 0, "errors": 0}

    if dry_run:
        logger.info("[DRY RUN] Aucun fichier ne sera écrit.")
        for src in all_images:
            # Chemin relatif de l'image par rapport au dossier raw
            rel = src.relative_to(input_dir)
            dst = output_dir / rel.with_suffix(".jpg")
            status = "EXISTS" if dst.exists() else "WOULD WRITE"
            logger.info(f"  {status} : {dst}")
        return counters

    # Traitement
    for src in tqdm(all_images, desc="Prétraitement images", unit="img"):
        rel = src.relative_to(input_dir)
        dst = (output_dir / rel).with_suffix(".jpg")

        # Ignorer si déjà traité et overwrite=False
        if dst.exists() and not overwrite:
            counters["skipped"] += 1
            continue

        ok = process_image(src, dst, max_size=max_size)
        if ok:
            counters["processed"] += 1
        else:
            counters["errors"] += 1

    logger.info(
        f"Terminé — "
        f"{counters['processed']} traitées, "
        f"{counters['skipped']} ignorées (déjà existantes), "
        f"{counters['errors']} erreurs."
    )
    return counters


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prétraite les images brutes (raw → processed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Répertoire source (raw/images par défaut).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Répertoire de sortie (processed/images par défaut).",
    )
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="Taille max en pixels (plus grande dimension). Pas de redim si absent.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simule sans écrire de fichiers.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Réécrit les fichiers déjà présents dans processed.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_prepare_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        max_size=args.max_size,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
