"""
Point d'entrée principal — Nutri-IA Data Collection.

Commandes disponibles
---------------------
    python main.py prepare        # Prétraite les images raw → processed
    python main.py prepare --help # Options complètes
    python main.py embed          # Génère les embeddings des images processed
    python main.py embed --help   # Options complètes
    python main.py info           # Affiche la config active et les stats modèle

Workflow typique
----------------
    python main.py prepare        # 1. raw → processed
    python main.py embed          # 2. processed → embeddings
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from src.config.settings import embed_cfg, paths


def cmd_prepare(args: argparse.Namespace) -> None:
    """Lance le pipeline de prétraitement raw → processed."""
    from src.process.prepare_images import run_prepare_pipeline

    counters = run_prepare_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        max_size=args.max_size,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(
        f"\nRésultat : {counters['processed']} traitées, "
        f"{counters['skipped']} ignorées, "
        f"{counters['errors']} erreurs "
        f"(sur {counters['total']} images sources)."
    )


def cmd_embed(args: argparse.Namespace) -> None:
    """Lance le pipeline de génération d'embeddings."""
    from src.process.embed_images import run_embedding_pipeline

    run_embedding_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        output_format=args.format,
        batch_size=args.batch_size,
        freeze_backbone=not args.no_freeze,
    )


def cmd_info(_args: argparse.Namespace) -> None:
    """Affiche la configuration active et un résumé du modèle."""
    from src.process.embedding_model import FoodEmbeddingModel

    print("\n=== Configuration active ===")
    for field, value in embed_cfg.model_dump().items():
        print(f"  {field:30s}: {value}")

    print("\n=== Résumé du modèle ===")
    model = FoodEmbeddingModel.from_config()
    print(model)
    counts = model.count_parameters()
    print(f"\n  Paramètres total      : {counts['total']:>12,}")
    print(f"  Paramètres entraîn.   : {counts['trainable']:>12,}")
    print(f"  Paramètres gelés      : {counts['frozen']:>12,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nutri-ia",
        description="Nutri-IA — Pipeline de collecte et traitement de données.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- prepare ---
    p_prepare = sub.add_parser(
        "prepare",
        help="Prétraite les images brutes (raw → processed).",
    )
    p_prepare.add_argument(
        "--input", type=str, default=None,
        help="Répertoire source (data/raw/images par défaut).",
    )
    p_prepare.add_argument(
        "--output", type=str, default=None,
        help="Répertoire de sortie (data/processed/images par défaut).",
    )
    p_prepare.add_argument(
        "--max-size", type=int, default=None,
        help="Taille max en pixels (plus grande dimension). Pas de redim si absent.",
    )
    p_prepare.add_argument(
        "--dry-run", action="store_true",
        help="Simule sans écrire de fichiers.",
    )
    p_prepare.add_argument(
        "--overwrite", action="store_true",
        help="Réécrit les fichiers déjà présents dans processed.",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    # --- embed ---
    p_embed = sub.add_parser("embed", help="Génère les embeddings d'images.")
    p_embed.add_argument("--input", type=str, default=None)
    p_embed.add_argument("--output", type=str, default=None)
    p_embed.add_argument("--checkpoint", type=str, default=None)
    p_embed.add_argument(
        "--format", choices=["hdf5", "npy", "parquet"], default=None
    )
    p_embed.add_argument("--batch-size", type=int, default=None)
    p_embed.add_argument("--no-freeze", action="store_true")
    p_embed.set_defaults(func=cmd_embed)

    # --- info ---
    p_info = sub.add_parser("info", help="Affiche la configuration et les stats du modèle.")
    p_info.set_defaults(func=cmd_info)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)
