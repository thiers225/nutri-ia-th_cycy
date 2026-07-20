# Évaluation des Embeddings d'Images — Nutri-IA

## Sommaire

1. [Pourquoi évaluer un modèle d'embedding ?](#pourquoi-évaluer)
2. [Métriques implémentées](#métriques-implémentées)
   - [k-NN Accuracy](#1-k-nn-accuracy)
   - [Precision@K et Recall@K](#2-precisionk-et-recallk)
   - [Mean Average Precision (mAP)](#3-mean-average-precision-map)
   - [Silhouette Score](#4-silhouette-score)
   - [Distances intra/inter-classes](#5-distances-intrainter-classes)
3. [Visualisations](#visualisations)
   - [Projection UMAP](#1-projection-umap)
   - [Heatmap des distances](#2-heatmap-des-distances-inter-classes)
   - [Plus proches voisins](#3-inspection-des-plus-proches-voisins)
4. [Interprétation des résultats](#interprétation-des-résultats)
5. [Utilisation](#utilisation)
   - [API Python](#api-python)
   - [CLI](#cli)
6. [Dépendances](#dépendances)
7. [Références](#références)

---

## Pourquoi évaluer ?

Un modèle d'embedding projette chaque image dans un espace vectoriel de dimension fixe.
La **qualité** de cet espace se mesure à sa capacité à :

- **rapprocher** les images visuellement et sémantiquement similaires (même plat),
- **éloigner** les images de plats différents,
- **généraliser** à des images inédites (non vues à l'entraînement).

Sans évaluation quantitative, il est impossible de savoir si le modèle capture la
sémantique alimentaire ou si ses vecteurs sont arbitraires. L'évaluation guide aussi
le choix du backbone, de la dimension d'embedding, et la décision de fine-tuner.

---

## Métriques implémentées

### 1. k-NN Accuracy

**Principe**
Pour chaque image de test, on cherche ses _k_ plus proches voisins dans l'espace
d'embedding (par distance cosinus) et on prédit la classe majoritaire parmi ces voisins.
L'accuracy mesure le taux de prédictions correctes.

**Pourquoi c'est pertinent**
Le k-NN ne suppose aucun apprentissage supplémentaire : il teste directement la
structure géométrique de l'espace d'embedding. Un bon modèle ne nécessite pas de
classifieur linéaire pour séparer les classes.

**Formule**

$$\text{k-NN Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\hat{y}_i^{(k)} = y_i\right]$$

où $\hat{y}_i^{(k)}$ est la classe prédite par vote majoritaire parmi les $k$ voisins.

**Validation croisée**
On utilise une **validation croisée stratifiée à 5 folds** pour obtenir une estimation
stable avec intervalle de confiance (moyenne ± écart-type). La stratification garantit
que chaque fold contient des représentants de toutes les classes.

**Interprétation**

| Résultat | Signification |
|---|---|
| ≈ 1/N_classes | Aussi bon que le hasard → embedding inutile |
| 0.5 – 0.7 | Structure partielle, amélioration possible |
| > 0.8 | Bon embedding pour la récupération |
| > 0.95 | Excellent, proche d'un oracle |

> **Référence :** Cover & Hart (1967) ont établi les propriétés théoriques du k-NN.
> Son utilisation comme proxy d'évaluation d'embedding est documentée dans
> [Kaya & Bilge (2019)](#références) et généralisée dans les benchmarks de
> représentation auto-supervisée comme [SimCLR (Chen et al., 2020)](#références).

---

### 2. Precision@K et Recall@K

**Principe**
Pour une image requête donnée, on récupère les _K_ images les plus proches dans
l'espace d'embedding. On mesure quelle fraction appartient à la même classe.

**Formules**

$$P@K = \frac{|\{\text{images pertinentes parmi les } K \text{ retournées}\}|}{K}$$

$$R@K = \frac{|\{\text{images pertinentes parmi les } K \text{ retournées}\}|}{|\{\text{images pertinentes au total}\}|}$$

**Pourquoi cette métrique**
C'est la métrique naturelle pour un **système de recherche d'images** (Image
Retrieval). Dans Nutri-IA, l'usage cible est de retrouver des images similaires
d'un même plat, ce qui correspond exactement à ce scénario.

**Valeurs typiques testées** : K ∈ {1, 5, 10}.

> **Référence :** Manning, Raghavan & Schütze, _Introduction to Information
> Retrieval_ (2008), Cambridge University Press — Chapitre 8 : Evaluation in
> information retrieval. Disponible en ligne :
> [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

---

### 3. Mean Average Precision (mAP)

**Principe**
La Average Precision (AP) pour une requête est l'aire sous la courbe
Precision-Recall. Le mAP est la moyenne des AP sur toutes les requêtes.

**Formule**

$$AP_i = \frac{1}{|R_i|} \sum_{k=1}^{N} P(k) \cdot \text{rel}(k)$$

$$\text{mAP} = \frac{1}{Q} \sum_{i=1}^{Q} AP_i$$

où $R_i$ est l'ensemble des documents pertinents pour la requête $i$,
$P(k)$ la précision au rang $k$, et $\text{rel}(k) = 1$ si le document au rang
$k$ est pertinent.

**Pourquoi le mAP**
Contrairement à P@K qui fixe un seuil arbitraire, le mAP évalue la qualité du
**classement complet** des résultats. Un mAP élevé garantit que les images
pertinentes apparaissent systématiquement en tête de liste.

> **Référence :** Zhu (2004), _Recall, Precision and Average Precision_,
> Department of Statistics and Actuarial Science, University of Waterloo.
> [https://cs.stanford.edu/people/ang/papers/maap.pdf](https://cs.stanford.edu/people/ang/papers/maap.pdf)
>
> Également : Musgrave, Belongie & Lim (2020), _A Metric Learning Reality Check_,
> ECCV 2020. [https://arxiv.org/abs/2003.08505](https://arxiv.org/abs/2003.08505)

---

### 4. Silhouette Score

**Principe**
Pour chaque point $i$, on calcule :
- $a(i)$ : distance moyenne à tous les autres points de **sa propre classe**
  (cohésion intra-classe, doit être petite)
- $b(i)$ : distance moyenne minimale aux points de la **classe la plus proche**
  (séparation inter-classes, doit être grande)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Le score global est la moyenne de $s(i)$ sur tous les points.

**Plage de valeurs**

| Score | Interprétation |
|---|---|
| 0.7 – 1.0 | Clusters très bien définis |
| 0.5 – 0.7 | Structure raisonnable |
| 0.25 – 0.5 | Structure faible |
| < 0.25 | Peu ou pas de structure |

**Note de distance** : on utilise la **distance cosinus** (1 - similarité cosinus),
cohérente avec la normalisation L2 du modèle.

> **Référence :** Rousseeuw, P.J. (1987), _Silhouettes: A graphical aid to the
> interpretation and validation of cluster analysis_, Journal of Computational
> and Applied Mathematics, 20, 53–65.
> [https://doi.org/10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)

---

### 5. Distances intra/inter-classes

**Principe**
On calcule directement les distances cosinus moyennes entre paires d'images :
- **Intra-classe** : paires $(i, j)$ telles que $y_i = y_j$ (même plat)
- **Inter-classes** : paires $(i, j)$ telles que $y_i \neq y_j$ (plats différents)

**Ratio de séparation**

$$\text{ratio} = \frac{\bar{d}_{\text{inter}}}{\bar{d}_{\text{intra}}}$$

Un ratio > 1 indique que les plats différents sont en moyenne plus éloignés
que les images du même plat. Plus le ratio est grand, mieux c'est.

**Lien avec la Triplet Loss**
Cette métrique mesure directement ce qu'optimise une fonction de perte de type
Triplet Loss ou Contrastive Loss :
$d(\text{ancre}, \text{positif}) \ll d(\text{ancre}, \text{négatif})$

> **Référence :** Schroff, Kalenichenko & Philbin (2015), _FaceNet: A Unified
> Embedding for Face Recognition and Clustering_, CVPR 2015.
> [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)

---

## Visualisations

### 1. Projection UMAP

**UMAP** (Uniform Manifold Approximation and Projection) réduit l'espace
d'embedding en 2D en préservant à la fois la structure locale (voisinage)
et globale (distances relatives entre clusters).

**Ce qu'on cherche à voir**
- Des **clusters distincts** par classe alimentaire → bon embedding
- Des **régions qui se chevauchent** entre classes visuellement similaires
  (ex. plats en sauce) → comportement attendu et acceptable
- Des **points isolés** au sein de leur cluster → images atypiques ou bruyantes

**Paramètres clés**
- `n_neighbors` : contrôle l'équilibre structure locale/globale (défaut : 15)
- `min_dist` : compacité des clusters dans la projection (défaut : 0.1)
- `metric="cosine"` : cohérent avec l'espace d'embedding normalisé L2

> **Référence :** McInnes, Healy & Melville (2018), _UMAP: Uniform Manifold
> Approximation and Projection for Dimension Reduction_,
> [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)

---

### 2. Heatmap des distances inter-classes

La heatmap montre la matrice des distances cosinus **moyennes** entre chaque
paire de classes. Elle permet d'identifier :

- Les classes **difficiles à distinguer** (cellule claire = proches dans l'espace)
- Les classes **bien séparées** (cellule foncée = éloignées)
- Les **confusions typiques** du modèle

La diagonale (distance intra-classe) doit être systématiquement plus foncée
que le reste de la ligne, signe que chaque classe est plus cohérente en interne
qu'avec n'importe quelle autre.

---

### 3. Inspection des plus proches voisins

La visualisation qualitative est irremplaçable : elle montre les **cas concrets**
où le modèle réussit ou échoue. Pour chaque image requête, on affiche ses
$k$ plus proches voisins avec un code couleur :

- **Vert** : voisin de la même classe → récupération correcte
- **Rouge** : voisin d'une classe différente → erreur de récupération

Cette inspection permet de diagnostiquer les causes d'erreur :
conditions d'éclairage, angle de vue, plats visuellement similaires, etc.

---

## Interprétation des résultats

### Grille de lecture globale

| Métrique | Mauvais | Acceptable | Bon | Excellent |
|---|---|---|---|---|
| k-NN Acc. (k=1) | ≈ 1/C | 0.4–0.6 | 0.7–0.85 | > 0.9 |
| mAP | < 0.3 | 0.3–0.5 | 0.5–0.75 | > 0.75 |
| Silhouette | < 0.1 | 0.1–0.25 | 0.25–0.5 | > 0.5 |
| Ratio inter/intra | < 1.0 | 1.0–1.3 | 1.3–2.0 | > 2.0 |

_C = nombre de classes. Les seuils sont indicatifs et dépendent du dataset._

### Que faire si les métriques sont mauvaises ?

**k-NN proche du hasard (1/C)**
→ Le modèle n'a pas appris de structure utile. Envisager un fine-tuning ou
changer de backbone.

**Silhouette négatif**
→ Les embeddings de classes différentes se mélangent. Problème de normalisation,
de dimension trop faible, ou backbone inadapté.

**mAP faible mais k-NN correct**
→ Le modèle classe bien mais ne range pas bien les résultats. Les images
pertinentes n'arrivent pas systématiquement en tête.

**UMAP : clusters qui se chevauchent fortement**
→ Certaines classes sont visuellement ambiguës. Inspecter les erreurs avec
`show_nearest_neighbors()` pour comprendre lesquelles.

---

## Utilisation

### API Python

```python
from src.evaluate.embedding_eval import EmbeddingEvaluator

# --- Chargement depuis HDF5 (format Nutri-IA par défaut) ---
evaluator = EmbeddingEvaluator.from_hdf5(
    "data/processed/embeddings/embeddings.h5"
)

# --- Rapport complet (toutes métriques) ---
report = evaluator.full_report(
    k_values=[1, 3, 5, 10],   # valeurs k pour le k-NN
    retrieval_k=[1, 5, 10],   # valeurs K pour P@K / R@K
    knn_folds=5,               # folds pour la validation croisée
)
print(report.summary())

# --- Sauvegarder le rapport JSON ---
evaluator.save_report(report, "reports/eval_report.json")

# --- Métriques individuelles ---
knn_results  = evaluator.knn_accuracy(k_values=[1, 5])
retrieval    = evaluator.retrieval_metrics(k_values=[1, 5, 10])
sil_score    = evaluator.silhouette()
dist_stats   = evaluator.intra_inter_distances()

# --- Visualisations ---
evaluator.plot_umap(save_path="reports/umap.png", show=False)
evaluator.plot_distance_heatmap(save_path="reports/heatmap.png", show=False)
evaluator.show_nearest_neighbors(n_queries=4, k=5, save_path="reports/neighbors.png")
```

**Chargement depuis NumPy :**

```python
evaluator = EmbeddingEvaluator.from_numpy(
    embeddings_path="data/processed/embeddings/embeddings.npy",
    paths_path="data/processed/embeddings/paths.npy",
    labels_path="data/processed/embeddings/labels.npy",  # optionnel
)
```

**Instanciation directe :**

```python
import numpy as np

embeddings = np.random.randn(200, 256).astype("float32")
labels = np.repeat(np.arange(10), 20)   # 10 classes, 20 images chacune

evaluator = EmbeddingEvaluator(
    embeddings=embeddings,
    labels=labels,
    class_names=["Kedjenou", "alloco", ...],
)
```

### CLI

Toutes les commandes s'exécutent **depuis la racine du projet**
(`/Users/tiesko/Documents/DAH/nutri`). Le préfixe `uv run` garantit que
l'environnement virtuel `.venv` du projet est utilisé.

```bash
# Évaluation complète — rapport JSON + toutes les visualisations
uv run python -m src.evaluate.embedding_eval \
    --embeddings data/processed/embeddings/embeddings.h5 \
    --output     reports/eval_report.json \
    --umap       reports/umap.png \
    --heatmap    reports/distance_heatmap.png \
    --neighbors  reports/nearest_neighbors.png
```

Le dossier `reports/` est créé automatiquement s'il n'existe pas.

```bash
# Sans visualisations — uniquement le rapport texte + JSON (plus rapide)
uv run python -m src.evaluate.embedding_eval \
    --embeddings data/processed/embeddings/embeddings.h5 \
    --output     reports/eval_report.json \
    --no-plots
```

```bash
# Valeurs de k personnalisées pour le k-NN et le retrieval
uv run python -m src.evaluate.embedding_eval \
    --embeddings  data/processed/embeddings/embeddings.h5 \
    --output      reports/eval_report.json \
    --k-values    1 5 10 20 \
    --retrieval-k 5 10 20
```

**Alternative** : si l'environnement virtuel est déjà activé
(`source .venv/bin/activate`), le préfixe `uv run` n'est pas nécessaire :

```bash
python -m src.evaluate.embedding_eval \
    --embeddings data/processed/embeddings/embeddings.h5 \
    --output     reports/eval_report.json
```

**Résultat attendu dans le terminal :**

```
============================================================
RAPPORT D'ÉVALUATION — Nutri-IA Embeddings
============================================================
  Échantillons   : 2000
  Classes        : 20
  Dim. embedding : 256
  Baseline (aléatoire) : 0.050

[ k-NN Accuracy ]
  k= 1 → 0.7823 ± 0.0142
  k= 3 → 0.7651 ± 0.0118
  k= 5 → 0.7490 ± 0.0103
  k=10 → 0.7112 ± 0.0097

[ Retrieval ]
  P@ 1 = 0.7823   R@ 1 = 0.0412
  P@ 5 = 0.7234   R@ 5 = 0.1902
  P@10 = 0.6891   R@10 = 0.3621
  mAP  = 0.7105

[ Clustering ]
  Silhouette Score  = 0.3241  (max=1, bon>0.2)
  Dist. intra-classe = 0.2134
  Dist. inter-classe = 0.6782
  Ratio séparation  = 3.1786  (>1 = bon)

  Durée totale : 42.3s
============================================================
```

---

## Dépendances

Les dépendances d'évaluation sont séparées des dépendances de production.
Ajouter avec `uv` :

```bash
# Dépendances principales (métriques)
uv add scikit-learn h5py

# Visualisations
uv add matplotlib umap-learn Pillow
```

| Package | Rôle | Version min. |
|---|---|---|
| `scikit-learn` | k-NN, silhouette, LabelEncoder | ≥ 1.3 |
| `h5py` | Lecture des fichiers HDF5 | ≥ 3.0 |
| `numpy` | Calculs matriciels | ≥ 1.24 |
| `umap-learn` | Projection UMAP | ≥ 0.5 |
| `matplotlib` | Visualisations | ≥ 3.7 |
| `Pillow` | Chargement des images | ≥ 10.0 |
| `loguru` | Journalisation | ≥ 0.7 |

---

## Références

### Métriques d'évaluation d'embeddings

1. **Cover, T. & Hart, P. (1967)**. _Nearest neighbor pattern classification_.
   IEEE Transactions on Information Theory, 13(1), 21–27.
   [https://doi.org/10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)
   > Fondements théoriques du classifieur k-NN.

2. **Rousseeuw, P.J. (1987)**. _Silhouettes: A graphical aid to the interpretation
   and validation of cluster analysis_. Journal of Computational and Applied
   Mathematics, 20, 53–65.
   [https://doi.org/10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)
   > Article original du Silhouette Score.

3. **Manning, C., Raghavan, P. & Schütze, H. (2008)**. _Introduction to
   Information Retrieval_. Cambridge University Press.
   [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)
   > Référence standard pour Precision@K, Recall@K et mAP (Chapitre 8).

4. **Musgrave, K., Belongie, S. & Lim, S.N. (2020)**. _A Metric Learning Reality
   Check_. ECCV 2020.
   [https://arxiv.org/abs/2003.08505](https://arxiv.org/abs/2003.08505)
   > Critique méthodologique et recommandations pour évaluer les espaces d'embedding.
   > Recommande fortement P@1 et mAP comme métriques primaires.

5. **Kaya, M. & Bilge, H.Ş. (2019)**. _Deep Metric Learning: A Survey_.
   Symmetry, 11(9), 1066.
   [https://doi.org/10.3390/sym11091066](https://doi.org/10.3390/sym11091066)
   > Tour d'horizon des méthodes et métriques d'évaluation en metric learning.

### Représentation visuelle et auto-supervision

6. **Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020)**. _A Simple
   Framework for Contrastive Learning of Visual Representations (SimCLR)_.
   ICML 2020.
   [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)
   > Utilise le k-NN accuracy (linear evaluation) comme métrique d'évaluation
   > standard pour les représentations auto-supervisées.

7. **Schroff, F., Kalenichenko, D. & Philbin, J. (2015)**. _FaceNet: A Unified
   Embedding for Face Recognition and Clustering_. CVPR 2015.
   [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)
   > Introduit le ratio intra/inter distances comme critère de qualité d'un
   > espace d'embedding.

8. **Caron, M., Touvron, H., Misra, I., et al. (2021)**. _Emerging Properties in
   Self-Supervised Vision Transformers (DINO)_. ICCV 2021.
   [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)
   > Démontre qu'un bon espace d'embedding permet la segmentation sémantique
   > sans supervision — la qualité se mesure par k-NN et retrieval.

### Réduction de dimension et visualisation

9. **McInnes, L., Healy, J. & Melville, J. (2018)**. _UMAP: Uniform Manifold
   Approximation and Projection for Dimension Reduction_.
   [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
   > Article fondateur de UMAP. Préféré à t-SNE pour les grands datasets et
   > la préservation de la structure globale.

10. **Van der Maaten, L. & Hinton, G. (2008)**. _Visualizing Data using t-SNE_.
    Journal of Machine Learning Research, 9, 2579–2605.
    [https://www.jmlr.org/papers/v9/vandermaaten08a.html](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
    > Alternative à UMAP pour la visualisation (meilleure préservation locale,
    > plus lente sur grands datasets).

### Reconnaissance d'images alimentaires

11. **Martinel, N., Foresti, G.L. & Micheloni, C. (2018)**. _Wide-Slice Residual
    Networks for Food Recognition_. IEEE Winter Conference on Applications of
    Computer Vision (WACV).
    [https://arxiv.org/abs/1612.06543](https://arxiv.org/abs/1612.06543)
    > Évaluation d'embeddings sur des datasets alimentaires ; montre l'importance
    > du fine-tuning pour les plats non-occidentaux.

12. **Waltner, G., Schwarz, M., Ladstätter, S., et al. (2017)**. _Personalized
    dietary self-management using mobile vision-based food recognition_.
    IEEE CVPR Workshop.
    > Applique les métriques de retrieval dans un contexte nutritionnel proche
    > de Nutri-IA.

---

_Document maintenu dans le dépôt Nutri-IA — dernière mise à jour : 2026-07-20_
