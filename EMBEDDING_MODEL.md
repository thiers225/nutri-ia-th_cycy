# Modèle d'Embedding d'Images — Nutri-IA

## Sommaire

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
   - [API Python](#api-python)
   - [CLI](#cli)
6. [Format des sorties](#format-des-sorties)
7. [Dataset & Prétraitement](#dataset--prétraitement)
8. [Sérialisation](#sérialisation)
9. [Choix de conception](#choix-de-conception)
10. [Référence des modules](#référence-des-modules)

---

## Vue d'ensemble

Le modèle d'embedding transforme une image alimentaire (photo de plat, étiquette nutritionnelle, ingrédient) en un vecteur dense de dimension fixe. Ces vecteurs sont ensuite utilisés pour :

- la **recherche par similarité** (trouver des aliments visuellement proches) ;
- l'**alimentation d'un modèle aval** (classification nutritionnelle, RAG visuel) ;
- la **déduplication** du dataset d'images.

---

## Architecture

```
Image RGB  (B × 3 × 224 × 224)
      │
┌─────▼──────────────────────────────┐
│  Backbone (timm)                   │
│  ex. EfficientNet-B2 → 1408d       │
│  ex. ResNet-50       → 2048d       │
│  ex. ViT-B/16        → 768d        │
│  global_pool="avg"  (GAP)          │
└─────┬──────────────────────────────┘
      │  features  (B × D_backbone)
┌─────▼──────────────────────────────┐
│  Projection Head (MLP)             │
│                                    │
│  Linear(D_backbone → hidden)       │
│  BatchNorm1d  →  ReLU  →  Dropout  │
│                                    │
│  Linear(hidden → hidden/2)         │
│  BatchNorm1d  →  ReLU  →  Dropout  │
│                                    │
│  Linear(hidden/2 → embedding_dim)  │
└─────┬──────────────────────────────┘
      │  embedding  (B × embedding_dim)
┌─────▼──────────────────────────────┐
│  L2 Normalize (optionnel)          │
│  ‖e‖₂ = 1                         │
└────────────────────────────────────┘
```

### Backbone

Le backbone est chargé via la bibliothèque **timm** avec `num_classes=0` (suppression de la tête de classification) et `global_pool="avg"` (Global Average Pooling). N'importe quel modèle timm peut être utilisé.

**Backbones recommandés :**

| Backbone | Dim features | Params | Vitesse CPU | Usage conseillé |
|---|---|---|---|---|
| `efficientnet_b2` | 1408 | 9M | ★★★ | Défaut — bon équilibre |
| `resnet50` | 2048 | 25M | ★★★ | Bonne baseline |
| `convnext_tiny` | 768 | 28M | ★★ | Performances élevées |
| `vit_base_patch16_224` | 768 | 86M | ★ | Max précision, GPU requis |
| `resnet18` | 512 | 11M | ★★★★ | Tests / dev rapide |

### Tête de projection

Un MLP à 3 couches réduit les features backbone vers un espace d'embedding de dimension configurable (défaut : **256**). L'utilisation d'une tête de projection plutôt que d'utiliser directement les features backbone permet :

- de contrôler la dimension finale indépendamment du backbone ;
- d'adapter l'espace sémantique au domaine alimentaire sans modifier le backbone ;
- de faciliter le fine-tuning partiel.

### Normalisation L2

Quand `normalize_output=True` (défaut), chaque vecteur est divisé par sa norme L2. Cela permet d'utiliser le **produit scalaire** comme mesure de similarité (équivalent à la cosine similarity sur vecteurs normalisés), ce qui est standard pour les systèmes de recherche par embedding.

---

## Configuration

Toute la configuration est centralisée dans `src/config/settings.py` via **Pydantic Settings**.

Chaque paramètre peut être surchargé par :
1. une variable d'environnement préfixée `NUTRIA_EMBED_` ;
2. un fichier `.env` à la racine du projet.

**Paramètres principaux :**

| Paramètre | Défaut | Description |
|---|---|---|
| `backbone` | `efficientnet_b2` | Architecture timm |
| `pretrained` | `True` | Poids ImageNet |
| `embedding_dim` | `256` | Dimension de l'embedding final |
| `projection_hidden_dim` | `512` | Dimension cachée de la projection |
| `projection_dropout` | `0.2` | Taux de dropout |
| `normalize_output` | `True` | Normalisation L2 |
| `image_size` | `224` | Taille d'entrée (px) |
| `batch_size` | `64` | Batch d'inférence |
| `device` | `auto` | cpu / cuda / mps / auto |
| `output_format` | `hdf5` | Format de sortie des embeddings |

**Exemple de fichier `.env` :**

```dotenv
NUTRIA_EMBED_BACKBONE=convnext_tiny
NUTRIA_EMBED_EMBEDDING_DIM=512
NUTRIA_EMBED_BATCH_SIZE=128
NUTRIA_EMBED_DEVICE=cuda
```

---

## Installation

Le projet utilise **uv** comme gestionnaire de paquets.

```bash
# Installer uv si nécessaire
pip install uv

# Créer l'environnement et installer les dépendances
uv sync

# Dépendances de développement (tests, linting)
uv sync --extra dev
```

---

## Utilisation

### API Python

#### Instantiation

```python
from src.process.embedding_model import FoodEmbeddingModel
from src.config.settings import embed_cfg

# Depuis la config globale (recommandé)
model = FoodEmbeddingModel.from_config()

# Depuis un checkpoint sauvegardé
model = FoodEmbeddingModel.load("models/embedding_model.pt")

# Manuel
model = FoodEmbeddingModel(
    backbone_name="resnet50",
    pretrained=True,
    embedding_dim=256,
    normalize_output=True,
)
```

#### Générer des embeddings

```python
import torch

# Batch d'images prétraitées (B, 3, H, W)
images = torch.randn(8, 3, 224, 224)

# Inférence (sans gradient, retourne numpy)
embeddings = model.encode(images, return_numpy=True)
print(embeddings.shape)  # (8, 256)

# Forward complet (pour l'entraînement)
model.train()
embeddings = model(images)  # (8, 256)
```

#### Features brutes du backbone

```python
# Avant la tête de projection (utile pour le fine-tuning)
features = model.extract_features(images)  # (8, 1408) pour EfficientNet-B2
```

#### Gel du backbone (transfer learning)

```python
# Geler tout le backbone, entraîner seulement la projection
model = FoodEmbeddingModel.from_config(freeze_backbone=True)

# Fine-tuning partiel : dégeler les 2 derniers blocs
model = FoodEmbeddingModel.from_config(
    freeze_backbone=True,
    unfreeze_last_n_blocks=2,
)

# Dégeler tout
model.unfreeze_backbone()
```

#### Pipeline complet sur un dossier

```python
from src.process.embed_images import run_embedding_pipeline

out_path = run_embedding_pipeline(
    input_dir="data/processed/images",
    output_dir="data/processed/embeddings",
    output_format="hdf5",
    batch_size=64,
)
print(f"Embeddings sauvegardés → {out_path}")
```

### CLI

```bash
# Générer les embeddings (config par défaut)
python main.py embed

# Options avancées
python main.py embed \
    --input  data/processed/images \
    --output data/processed/embeddings \
    --checkpoint models/embedding_model.pt \
    --format hdf5 \
    --batch-size 32

# Afficher la config active et les stats du modèle
python main.py info
```

---

## Format des sorties

### HDF5 (défaut, recommandé)

Fichier `embeddings.h5` avec :

| Dataset | Type | Shape | Description |
|---|---|---|---|
| `embeddings` | float32 | (N, D) | Vecteurs d'embedding |
| `paths` | string | (N,) | Chemins absolus des images |
| `labels` | int32 | (N,) | Indices de classe (si disponibles) |

Attributs du fichier : `embedding_dim`, `n_samples`.

```python
import h5py, numpy as np

with h5py.File("data/processed/embeddings/embeddings.h5", "r") as f:
    embeddings = f["embeddings"][:]   # np.ndarray (N, 256)
    paths      = f["paths"][:]        # liste de strings
    labels     = f["labels"][:]       # np.ndarray (N,) — si présent
```

### NumPy

Trois fichiers : `embeddings.npy`, `paths.npy`, `labels.npy` (optionnel).

```python
embeddings = np.load("data/processed/embeddings/embeddings.npy")
paths      = np.load("data/processed/embeddings/paths.npy", allow_pickle=True)
```

### Parquet

Fichier `embeddings.parquet` avec colonnes `e_0`, `e_1`, …, `e_{D-1}`, `path`, `label` (optionnel).

```python
import pandas as pd

df = pd.read_parquet("data/processed/embeddings/embeddings.parquet")
# Reconstruire la matrice d'embeddings
embed_cols = [c for c in df.columns if c.startswith("e_")]
embeddings = df[embed_cols].values  # (N, D)
```

---

## Dataset & Prétraitement

### FoodImageDataset

Supporte deux structures de répertoire :

**Structure hiérarchique (avec labels) :**
```
data/processed/images/
├── pizza/
│   ├── img_001.jpg
│   └── img_002.jpg
├── salade/
│   └── img_003.jpg
└── ...
```

**Structure plate (sans labels) :**
```
data/processed/images/
├── img_001.jpg
├── img_002.jpg
└── ...
```

```python
from src.utils.image_utils import FoodImageDataset

dataset = FoodImageDataset(root_dir="data/processed/images")
print(dataset.classes)       # ['pizza', 'salade', ...]
print(dataset.class_to_idx)  # {'pizza': 0, 'salade': 1, ...}
```

### Transformations

| Pipeline | Usage | Augmentations |
|---|---|---|
| `get_inference_transforms()` | Génération d'embeddings, eval | Resize + CenterCrop + Normalize |
| `get_train_transforms()` | Fine-tuning | RandomResizedCrop + Flip + ColorJitter + Normalize |

Les valeurs de normalisation ImageNet sont utilisées par défaut (mean/std RGB).

---

## Sérialisation

```python
# Sauvegarder
model.save("models/embedding_model.pt")

# Charger
model = FoodEmbeddingModel.load("models/embedding_model.pt")
```

Le checkpoint `.pt` contient :
- `model_state_dict` — poids complets
- `backbone_name` — nom du backbone pour reconstruction
- `embedding_dim` — dimension de l'embedding
- `normalize_output` — flag de normalisation

---

## Choix de conception

**Pourquoi timm ?**
timm (PyTorch Image Models) offre un catalogue unifié de +700 architectures avec des poids pré-entraînés et une API cohérente. Changer de backbone revient à modifier une seule chaîne de caractères dans la config.

**Pourquoi une tête de projection plutôt que les features brutes ?**
Les features brutes d'un backbone ImageNet ne sont pas optimisées pour les aliments. La tête de projection est la partie entraînable qui adapte l'espace sémantique au domaine, sans nécessiter de fine-tuning coûteux du backbone entier.

**Pourquoi la normalisation L2 ?**
Elle permet d'utiliser le produit scalaire (dot product) pour la similarité, qui est nettement plus rapide que la distance euclidienne pour la recherche dans de grands catalogues (FAISS, etc.).

**Pourquoi HDF5 par défaut ?**
HDF5 supporte la compression, le chargement partiel (slicing), et stocke les métadonnées. C'est le format standard pour les grands datasets de vecteurs en ML.

---

## Référence des modules

| Module | Rôle |
|---|---|
| `src/config/settings.py` | Config centralisée (Pydantic Settings) |
| `src/process/embedding_model.py` | `FoodEmbeddingModel`, `ProjectionHead`, `resolve_device` |
| `src/process/embed_images.py` | `EmbeddingGenerator`, `run_embedding_pipeline` (CLI) |
| `src/utils/image_utils.py` | `FoodImageDataset`, `get_train_transforms`, `get_inference_transforms` |
| `tests/test_embedding_model.py` | Suite de tests unitaires |
| `main.py` | Point d'entrée CLI (`embed`, `info`) |
