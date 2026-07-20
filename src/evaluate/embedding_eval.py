"""
Évaluation des embeddings d'images — Nutri-IA.

Ce module fournit un ensemble de métriques et visualisations pour juger
la qualité de l'espace d'embedding produit par FoodEmbeddingModel :

  - k-NN Accuracy            : précision de classification au plus proche voisin
  - Retrieval Precision@K    : qualité de récupération des images similaires
  - Mean Average Precision   : mAP pour la récupération
  - Silhouette Score         : séparabilité intra/inter-classes
  - Intra/Inter distances    : distances moyennes par classe
  - Visualisation UMAP       : projection 2D de l'espace d'embedding

Utilisation rapide
------------------
>>> from src.evaluate.embedding_eval import EmbeddingEvaluator
>>> evaluator = EmbeddingEvaluator.from_hdf5("data/processed/embeddings/embeddings.h5")
>>> report = evaluator.full_report()
>>> evaluator.plot_umap(save_path="reports/umap.png")

CLI
---
    python -m src.evaluate.embedding_eval \\
        --embeddings data/processed/embeddings/embeddings.h5 \\
        --output     reports/eval_report.json

Références
----------
Voir EMBEDDING_EVALUATION.md pour la bibliographie complète.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from loguru import logger
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class KNNResult:
    """Résultats de l'évaluation k-NN."""
    k: int
    accuracy_mean: float
    accuracy_std: float
    accuracy_per_fold: list[float]


@dataclass
class RetrievalResult:
    """Résultats de récupération par similarité."""
    precision_at_k: dict[int, float]   # {k: précision moyenne}
    mean_average_precision: float
    recall_at_k: dict[int, float]      # {k: rappel moyen}


@dataclass
class DistanceResult:
    """Statistiques de distances intra/inter-classes."""
    intra_class_mean: float
    inter_class_mean: float
    separation_ratio: float            # inter / intra (plus grand = mieux)


@dataclass
class EvaluationReport:
    """Rapport complet d'évaluation d'un espace d'embedding."""
    n_samples: int
    n_classes: int
    embedding_dim: int
    knn_results: list[KNNResult]
    retrieval: RetrievalResult
    silhouette: float
    distances: DistanceResult
    random_baseline: float             # 1 / n_classes
    duration_seconds: float
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Retourne un résumé lisible du rapport."""
        best_knn = max(self.knn_results, key=lambda r: r.accuracy_mean)
        lines = [
            "=" * 60,
            "RAPPORT D'ÉVALUATION — Nutri-IA Embeddings",
            "=" * 60,
            f"  Échantillons   : {self.n_samples}",
            f"  Classes        : {self.n_classes}",
            f"  Dim. embedding : {self.embedding_dim}",
            f"  Baseline (aléatoire) : {self.random_baseline:.3f}",
            "",
            "[ k-NN Accuracy ]",
        ]
        for r in self.knn_results:
            lines.append(
                f"  k={r.k:2d} → {r.accuracy_mean:.4f} ± {r.accuracy_std:.4f}"
            )
        lines += [
            "",
            "[ Retrieval ]",
        ]
        for k, p in sorted(self.retrieval.precision_at_k.items()):
            lines.append(f"  P@{k:2d} = {p:.4f}   R@{k:2d} = {self.retrieval.recall_at_k[k]:.4f}")
        lines += [
            f"  mAP  = {self.retrieval.mean_average_precision:.4f}",
            "",
            "[ Clustering ]",
            f"  Silhouette Score  = {self.silhouette:.4f}  (max=1, bon>0.2)",
            f"  Dist. intra-classe = {self.distances.intra_class_mean:.4f}",
            f"  Dist. inter-classe = {self.distances.inter_class_mean:.4f}",
            f"  Ratio séparation  = {self.distances.separation_ratio:.4f}  (>1 = bon)",
            "",
            f"  Durée totale : {self.duration_seconds:.1f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def _cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de distances cosinus (1 - similarité) pour N vecteurs.

    On suppose les vecteurs déjà normalisés L2 (comme la sortie du modèle).
    Dans ce cas  dist = 1 - X @ X.T  est exact et très rapide.

    Args:
        X: Matrice (N, D) de vecteurs normalisés.

    Returns:
        Matrice symétrique (N, N) de distances ∈ [0, 2].
    """
    # Si les vecteurs ne sont pas normalisés, on normalise ici
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def _average_precision_query(
    sorted_labels: np.ndarray,
    query_label: int,
) -> float:
    """
    Calcule la Average Precision (AP) pour une requête donnée.

    Args:
        sorted_labels: Labels des voisins triés par distance croissante
                       (le premier est le voisin le plus proche).
        query_label: Label de la requête (exclu du calcul).

    Returns:
        AP ∈ [0, 1].
    """
    relevant = (sorted_labels == query_label)
    n_relevant_total = relevant.sum()
    if n_relevant_total == 0:
        return 0.0

    precisions = []
    n_retrieved = 0
    for rank, rel in enumerate(relevant, start=1):
        if rel:
            n_retrieved += 1
            precisions.append(n_retrieved / rank)

    return float(np.mean(precisions))


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class EmbeddingEvaluator:
    """
    Évalue la qualité d'un espace d'embedding sur un jeu d'images labellisées.

    Args:
        embeddings: Matrice (N, D) d'embeddings.
        labels: Vecteur (N,) d'indices de classes entiers.
        class_names: Liste des noms de classes (optionnel, pour affichage).
        paths: Chemins des images correspondantes (optionnel).

    Example:
        >>> ev = EmbeddingEvaluator(embeddings, labels, class_names)
        >>> report = ev.full_report()
        >>> print(report.summary())
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[list[str]] = None,
        paths: Optional[list[str]] = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings doit être 2D, reçu {embeddings.ndim}D.")
        if len(labels) != len(embeddings):
            raise ValueError("embeddings et labels doivent avoir la même longueur.")

        self.embeddings = embeddings.astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int32)
        self.class_names = class_names or [str(i) for i in np.unique(self.labels)]
        self.image_paths = paths
        self.n_samples, self.embedding_dim = self.embeddings.shape
        self.n_classes = len(np.unique(self.labels))

        logger.info(
            f"EmbeddingEvaluator — {self.n_samples} échantillons, "
            f"{self.n_classes} classes, dim={self.embedding_dim}"
        )

    # ------------------------------------------------------------------
    # Chargement depuis fichiers
    # ------------------------------------------------------------------

    @classmethod
    def from_hdf5(cls, hdf5_path: str | Path) -> "EmbeddingEvaluator":
        """
        Charge les embeddings depuis un fichier HDF5 (format Nutri-IA).

        Le fichier doit contenir les datasets ``embeddings``, ``paths``,
        et optionnellement ``labels``. Les labels sont inférés depuis
        les chemins si le dataset ``labels`` est absent.

        Args:
            hdf5_path: Chemin vers ``embeddings.h5``.

        Returns:
            Instance ``EmbeddingEvaluator`` prête à l'emploi.

        Raises:
            ValueError: Si aucun label ne peut être inféré.
        """
        path = Path(hdf5_path)
        logger.info(f"Chargement des embeddings depuis {path}")

        with h5py.File(path, "r") as f:
            embeddings = f["embeddings"][:]
            raw_paths = [
                p.decode("utf-8") if isinstance(p, bytes) else str(p)
                for p in f["paths"][:]
            ]
            if "labels" in f:
                labels_raw = f["labels"][:]
                # Si tous les labels sont -1 ou manquants, on infère depuis les chemins
                if np.all(labels_raw == -1) or np.all(labels_raw < 0):
                    labels_raw = None
                    logger.warning("Labels HDF5 tous négatifs — inférence depuis les chemins.")
            else:
                labels_raw = None

        # Inférence des labels depuis le répertoire parent du chemin
        if labels_raw is None:
            class_dirs = [Path(p).parent.name for p in raw_paths]
            if len(set(class_dirs)) <= 1:
                raise ValueError(
                    "Impossible d'inférer les classes : toutes les images sont "
                    "dans le même répertoire. Fournissez un fichier HDF5 avec labels."
                )
            le = LabelEncoder()
            labels_raw = le.fit_transform(class_dirs)
            class_names = list(le.classes_)
            logger.info(f"Labels inférés depuis les chemins : {class_names}")
        else:
            labels_raw = labels_raw.astype(np.int32)
            unique_labels = sorted(np.unique(labels_raw))
            class_names = [str(i) for i in unique_labels]

        return cls(
            embeddings=embeddings,
            labels=labels_raw,
            class_names=class_names,
            paths=raw_paths,
        )

    @classmethod
    def from_numpy(
        cls,
        embeddings_path: str | Path,
        paths_path: str | Path,
        labels_path: Optional[str | Path] = None,
    ) -> "EmbeddingEvaluator":
        """
        Charge les embeddings depuis des fichiers NumPy (.npy).

        Args:
            embeddings_path: Chemin vers ``embeddings.npy``.
            paths_path: Chemin vers ``paths.npy``.
            labels_path: Chemin vers ``labels.npy`` (optionnel).

        Returns:
            Instance ``EmbeddingEvaluator``.
        """
        embeddings = np.load(embeddings_path)
        raw_paths = np.load(paths_path, allow_pickle=True).tolist()

        if labels_path and Path(labels_path).exists():
            labels = np.load(labels_path).astype(np.int32)
            class_names = [str(i) for i in sorted(np.unique(labels))]
        else:
            class_dirs = [Path(p).parent.name for p in raw_paths]
            le = LabelEncoder()
            labels = le.fit_transform(class_dirs).astype(np.int32)
            class_names = list(le.classes_)

        return cls(embeddings=embeddings, labels=labels, class_names=class_names, paths=raw_paths)


    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------

    def knn_accuracy(
        self,
        k_values: list[int] | None = None,
        n_folds: int = 5,
    ) -> list[KNNResult]:
        """
        Évalue la précision de classification par k plus proches voisins (k-NN).

        Utilise une validation croisée stratifiée pour une estimation robuste.
        La métrique mesure si les k voisins les plus proches appartiennent à
        la même classe que la requête.

        Args:
            k_values: Liste de valeurs k à tester. Défaut : [1, 3, 5, 10].
            n_folds: Nombre de folds pour la validation croisée.

        Returns:
            Liste de ``KNNResult`` pour chaque k.
        """
        k_values = k_values or [1, 3, 5, 10]
        # Limiter k au nombre d'échantillons par classe - 1
        min_class_count = min(np.bincount(self.labels))
        k_values = [k for k in k_values if k < min_class_count]
        if not k_values:
            k_values = [1]

        results = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for k in k_values:
            fold_scores = []
            for train_idx, test_idx in skf.split(self.embeddings, self.labels):
                clf = KNeighborsClassifier(
                    n_neighbors=k,
                    metric="cosine",
                    algorithm="brute",
                    n_jobs=-1,
                )
                clf.fit(self.embeddings[train_idx], self.labels[train_idx])
                score = clf.score(self.embeddings[test_idx], self.labels[test_idx])
                fold_scores.append(score)

            results.append(KNNResult(
                k=k,
                accuracy_mean=float(np.mean(fold_scores)),
                accuracy_std=float(np.std(fold_scores)),
                accuracy_per_fold=fold_scores,
            ))
            logger.info(f"k-NN k={k}: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

        return results

    def retrieval_metrics(
        self,
        k_values: list[int] | None = None,
        max_queries: int = 500,
    ) -> RetrievalResult:
        """
        Calcule les métriques de récupération : Precision@K, Recall@K, mAP.

        Pour chaque image requête, on cherche ses K plus proches voisins
        dans l'espace d'embedding (en excluant l'image elle-même) et on
        vérifie combien appartiennent à la même classe.

        Args:
            k_values: Valeurs de K pour P@K et R@K. Défaut : [1, 5, 10].
            max_queries: Nombre maximum de requêtes (sous-échantillonnage
                         si N > max_queries pour accélérer le calcul).

        Returns:
            ``RetrievalResult`` avec les métriques agrégées.
        """
        k_values = k_values or [1, 5, 10]
        max_k = max(k_values)

        # Sous-échantillonnage stratifié si nécessaire
        if self.n_samples > max_queries:
            rng = np.random.default_rng(42)
            query_idx = rng.choice(self.n_samples, size=max_queries, replace=False)
        else:
            query_idx = np.arange(self.n_samples)

        dist_matrix = _cosine_distance_matrix(self.embeddings)

        precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
        recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        ap_scores: list[float] = []

        for q in query_idx:
            q_label = self.labels[q]
            # Trier les autres images par distance (on exclut q lui-même)
            dists = dist_matrix[q].copy()
            dists[q] = np.inf  # exclure l'image elle-même
            sorted_idx = np.argsort(dists)
            sorted_labels = self.labels[sorted_idx]

            # Nombre total d'images de la même classe (hors requête)
            n_relevant = int((self.labels == q_label).sum()) - 1

            # AP
            ap = _average_precision_query(sorted_labels[:max(self.n_samples, max_k)], q_label)
            ap_scores.append(ap)

            for k in k_values:
                top_k_labels = sorted_labels[:k]
                n_relevant_in_k = int((top_k_labels == q_label).sum())
                precision_sums[k] += n_relevant_in_k / k
                recall_sums[k] += (n_relevant_in_k / n_relevant) if n_relevant > 0 else 0.0

        n_q = len(query_idx)
        return RetrievalResult(
            precision_at_k={k: precision_sums[k] / n_q for k in k_values},
            recall_at_k={k: recall_sums[k] / n_q for k in k_values},
            mean_average_precision=float(np.mean(ap_scores)),
        )


    def silhouette(self, sample_size: int = 2000) -> float:
        """
        Calcule le Silhouette Score de l'espace d'embedding.

        Score ∈ [-1, 1] :
          - proche de  1 : classes bien séparées, compactes.
          - proche de  0 : classes qui se chevauchent.
          - proche de -1 : mauvais clustering, images mal assignées.

        Args:
            sample_size: Sous-échantillon pour accélérer le calcul si N > sample_size.

        Returns:
            Score silhouette moyen (cosine).
        """
        if self.n_samples > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(self.n_samples, size=sample_size, replace=False)
            X, y = self.embeddings[idx], self.labels[idx]
        else:
            X, y = self.embeddings, self.labels

        score = silhouette_score(X, y, metric="cosine", random_state=42)
        logger.info(f"Silhouette Score (cosine) = {score:.4f}")
        return float(score)

    def intra_inter_distances(self) -> DistanceResult:
        """
        Calcule les distances moyennes intra-classe et inter-classes.

        - **Intra-classe** : distance moyenne entre paires d'images du même plat.
          Doit être petite → le modèle rapproche les images similaires.
        - **Inter-classes** : distance moyenne entre paires d'images de plats différents.
          Doit être grande → le modèle éloigne les classes différentes.

        Returns:
            ``DistanceResult`` avec les statistiques de distances.
        """
        dist_matrix = _cosine_distance_matrix(self.embeddings)
        intra_dists = []
        inter_dists = []

        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                d = dist_matrix[i, j]
                if self.labels[i] == self.labels[j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)

            # Pour de grands datasets, on sous-échantillonne
            if i > 500:
                break

        intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0
        inter_mean = float(np.mean(inter_dists)) if inter_dists else 0.0
        ratio = (inter_mean / intra_mean) if intra_mean > 0 else float("inf")

        logger.info(
            f"Dist. intra={intra_mean:.4f}  inter={inter_mean:.4f}  ratio={ratio:.4f}"
        )
        return DistanceResult(
            intra_class_mean=intra_mean,
            inter_class_mean=inter_mean,
            separation_ratio=ratio,
        )

    # ------------------------------------------------------------------
    # Rapport complet
    # ------------------------------------------------------------------

    def full_report(
        self,
        k_values: list[int] | None = None,
        retrieval_k: list[int] | None = None,
        knn_folds: int = 5,
    ) -> EvaluationReport:
        """
        Génère le rapport d'évaluation complet.

        Enchaîne toutes les métriques dans l'ordre et retourne un
        ``EvaluationReport`` sérialisable en JSON.

        Args:
            k_values: Valeurs k pour le k-NN. Défaut : [1, 3, 5, 10].
            retrieval_k: Valeurs K pour P@K/R@K. Défaut : [1, 5, 10].
            knn_folds: Nombre de folds pour la validation croisée k-NN.

        Returns:
            ``EvaluationReport`` avec toutes les métriques calculées.
        """
        t0 = time.perf_counter()
        logger.info("--- Démarrage évaluation complète ---")

        knn = self.knn_accuracy(k_values=k_values, n_folds=knn_folds)
        retrieval = self.retrieval_metrics(k_values=retrieval_k)
        sil = self.silhouette()
        dist = self.intra_inter_distances()

        report = EvaluationReport(
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            embedding_dim=self.embedding_dim,
            knn_results=knn,
            retrieval=retrieval,
            silhouette=sil,
            distances=dist,
            random_baseline=1.0 / self.n_classes,
            duration_seconds=time.perf_counter() - t0,
            metadata={"class_names": self.class_names},
        )
        logger.info("--- Évaluation terminée ---")
        return report

    def save_report(self, report: EvaluationReport, output_path: str | Path) -> None:
        """
        Sauvegarde le rapport d'évaluation en JSON.

        Args:
            report: Rapport à sauvegarder.
            output_path: Chemin du fichier de sortie (.json).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        def _to_dict(obj):
            if hasattr(obj, "__dict__"):
                return {k: _to_dict(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_dict(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        with open(out, "w", encoding="utf-8") as f:
            json.dump(_to_dict(report), f, indent=2, ensure_ascii=False)
        logger.info(f"Rapport sauvegardé → {out}")


    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_umap(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        save_path: Optional[str | Path] = None,
        show: bool = True,
        figsize: tuple[float, float] = (12.0, 9.0),
    ) -> None:
        """
        Visualise l'espace d'embedding via une projection UMAP 2D.

        Chaque point représente une image, coloré par classe. Une bonne
        projection montre des clusters distincts par plat alimentaire.

        Args:
            n_neighbors: Voisins considérés par UMAP (contrôle structure locale).
                         Valeurs typiques : 5–50.
            min_dist: Distance minimale entre points dans la projection.
                      Petite valeur → clusters denses.
            n_components: Dimension de la projection (2 ou 3).
            save_path: Si fourni, sauvegarde la figure à ce chemin.
            show: Si ``True``, affiche la figure à l'écran.
            figsize: Taille de la figure en pouces.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import umap as umap_lib
        except ImportError as e:
            logger.error(
                f"Dépendances manquantes : {e}. "
                "Installez avec : uv add umap-learn matplotlib"
            )
            return

        logger.info("Calcul de la projection UMAP…")
        reducer = umap_lib.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric="cosine",
            random_state=42,
            verbose=False,
        )
        proj = reducer.fit_transform(self.embeddings)

        # Palette de couleurs
        colors = cm.get_cmap("tab20", self.n_classes)
        fig, ax = plt.subplots(figsize=figsize)

        for cls_idx, cls_name in enumerate(self.class_names):
            mask = self.labels == cls_idx
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[colors(cls_idx)],
                label=cls_name,
                s=18,
                alpha=0.75,
                linewidths=0,
            )

        ax.set_title(
            f"UMAP — Espace d'embedding Nutri-IA\n"
            f"({self.n_samples} images · {self.n_classes} classes · dim={self.embedding_dim})",
            fontsize=13,
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
            framealpha=0.8,
            markerscale=1.5,
        )
        plt.tight_layout()

        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            logger.info(f"Figure UMAP sauvegardée → {out}")
        if show:
            plt.show()
        plt.close()

    def plot_distance_heatmap(
        self,
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> None:
        """
        Affiche une heatmap des distances inter-classes moyennes.

        Chaque cellule (i, j) représente la distance cosinus moyenne entre
        les embeddings de la classe i et de la classe j. La diagonale
        (intra-classe) doit être la plus sombre (distance minimale).

        Args:
            save_path: Chemin de sauvegarde de la figure (optionnel).
            show: Afficher la figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib requis. Installez avec : uv add matplotlib")
            return

        n = self.n_classes
        heatmap = np.zeros((n, n), dtype=np.float32)

        dist_matrix = _cosine_distance_matrix(self.embeddings)

        for i in range(n):
            for j in range(n):
                mask_i = self.labels == i
                mask_j = self.labels == j
                block = dist_matrix[np.ix_(mask_i, mask_j)]
                if i == j:
                    # Exclure la diagonale (distance à soi-même)
                    idx = np.triu_indices(block.shape[0], k=1)
                    heatmap[i, j] = float(block[idx].mean()) if len(idx[0]) > 0 else 0.0
                else:
                    heatmap[i, j] = float(block.mean())

        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
        im = ax.imshow(heatmap, cmap="YlOrRd_r", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(self.class_names, fontsize=9)
        ax.set_title("Distances inter-classes moyennes (cosinus)\n(plus foncé = plus proche)", fontsize=12)
        plt.colorbar(im, ax=ax, label="Distance cosinus")
        plt.tight_layout()

        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            logger.info(f"Heatmap sauvegardée → {out}")
        if show:
            plt.show()
        plt.close()

    def show_nearest_neighbors(
        self,
        n_queries: int = 4,
        k: int = 5,
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> None:
        """
        Affiche N images requêtes avec leurs K plus proches voisins.

        Permet d'inspecter visuellement les erreurs de récupération.
        Un contour vert indique un voisin de la bonne classe,
        rouge une erreur.

        Args:
            n_queries: Nombre d'images requêtes à afficher.
            k: Nombre de voisins à afficher par requête.
            save_path: Chemin de sauvegarde (optionnel).
            show: Afficher la figure.
        """
        if self.image_paths is None:
            logger.warning("Chemins d'images non disponibles — impossible d'afficher les voisins.")
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from PIL import Image
        except ImportError:
            logger.error("matplotlib et Pillow requis.")
            return

        dist_matrix = _cosine_distance_matrix(self.embeddings)

        # Sélectionner des requêtes représentatives (une par classe si possible)
        rng = np.random.default_rng(42)
        class_indices = {
            c: np.where(self.labels == c)[0] for c in range(self.n_classes)
        }
        query_indices = []
        for c in range(self.n_classes):
            if len(query_indices) >= n_queries:
                break
            if len(class_indices[c]) > 0:
                query_indices.append(rng.choice(class_indices[c]))

        n_cols = k + 1  # colonne requête + k voisins
        fig, axes = plt.subplots(
            len(query_indices), n_cols,
            figsize=(2.5 * n_cols, 2.5 * len(query_indices)),
        )
        if len(query_indices) == 1:
            axes = axes[np.newaxis, :]

        for row, q_idx in enumerate(query_indices):
            q_label = self.labels[q_idx]
            dists = dist_matrix[q_idx].copy()
            dists[q_idx] = np.inf
            sorted_idx = np.argsort(dists)[:k]

            # Colonne 0 : requête
            ax = axes[row, 0]
            _imshow_safe(ax, self.image_paths[q_idx])
            ax.set_title(
                f"[REQUÊTE]\n{self.class_names[q_label]}",
                fontsize=7, color="black", fontweight="bold",
            )
            ax.axis("off")

            # Colonnes 1..k : voisins
            for col, nb_idx in enumerate(sorted_idx, start=1):
                ax = axes[row, col]
                nb_label = self.labels[nb_idx]
                color = "green" if nb_label == q_label else "red"
                _imshow_safe(ax, self.image_paths[nb_idx])
                ax.set_title(
                    f"k={col}\n{self.class_names[nb_label]}\nd={dist_matrix[q_idx, nb_idx]:.3f}",
                    fontsize=6.5, color=color,
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
                ax.axis("off")

        plt.suptitle(
            f"Plus proches voisins — {k}-NN (vert=correct, rouge=erreur)",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()

        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            logger.info(f"Figure voisins sauvegardée → {out}")
        if show:
            plt.show()
        plt.close()


def _imshow_safe(ax, path: str) -> None:
    """Affiche une image dans un axe matplotlib, avec fallback si corrompue."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=20, color="gray")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Évalue la qualité des embeddings d'images Nutri-IA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings",
        default="data/processed/embeddings/embeddings.h5",
        help="Chemin vers le fichier HDF5 d'embeddings.",
    )
    parser.add_argument(
        "--output",
        default="reports/eval_report.json",
        help="Chemin de sortie du rapport JSON.",
    )
    parser.add_argument(
        "--umap",
        default="reports/umap.png",
        help="Chemin de sortie de la visualisation UMAP.",
    )
    parser.add_argument(
        "--heatmap",
        default="reports/distance_heatmap.png",
        help="Chemin de sortie de la heatmap des distances.",
    )
    parser.add_argument(
        "--neighbors",
        default="reports/nearest_neighbors.png",
        help="Chemin de sortie de la figure des plus proches voisins.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Désactiver la génération des visualisations.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="Valeurs k pour le k-NN.",
    )
    parser.add_argument(
        "--retrieval-k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Valeurs K pour Precision@K et Recall@K.",
    )
    return parser


def main() -> None:
    """Point d'entrée principal pour l'évaluation en ligne de commande."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    evaluator = EmbeddingEvaluator.from_hdf5(args.embeddings)

    # Rapport textuel et JSON
    report = evaluator.full_report(
        k_values=args.k_values,
        retrieval_k=args.retrieval_k,
    )
    print(report.summary())
    evaluator.save_report(report, args.output)

    # Visualisations
    if not args.no_plots:
        evaluator.plot_umap(save_path=args.umap, show=False)
        evaluator.plot_distance_heatmap(save_path=args.heatmap, show=False)
        evaluator.show_nearest_neighbors(save_path=args.neighbors, show=False)

    logger.info("Évaluation complète. Rapport : " + args.output)


if __name__ == "__main__":
    main()
