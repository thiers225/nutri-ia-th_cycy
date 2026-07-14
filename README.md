# Nutri-IA — Image Embedding Pipeline

> Collecte, traitement et vectorisation d'images alimentaires pour l'Assistant IA de Nutrition.

---

## Table des matières

- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Architecture du projet](#architecture-du-projet)
- [Comprendre le code](#comprendre-le-code)
  - [1. Configuration — `src/config/settings.py`](#1-configuration--srcconfigsettingspy)
  - [2. Préparation des images — `src/utils/image_utils.py`](#2-préparation-des-images--srcutilsimage_utilspy)
  - [3. Le modèle — `src/process/embedding_model.py`](#3-le-modèle--srcprocessembedding_modelpy)
  - [4. Le pipeline — `src/process/embed_images.py`](#4-le-pipeline--srcprocessembed_imagespy)
  - [5. Point d'entrée — `main.py`](#5-point-dentrée--mainpy)
- [Flux complet d'une exécution](#flux-complet-dune-exécution)
- [Choix technologiques](#choix-technologiques)
- [Tests](#tests)

---

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) comme gestionnaire de paquets.

```bash
# Installer uv
pip install uv

# Installer les dépendances
uv sync

# Avec les dépendances de développement (tests, linting)
uv sync --extra dev
```

---

## Démarrage rapide

```bash
# Générer les embeddings des images dans data/processed/images/
python main.py embed

# Voir la configuration active et les stats du modèle
python main.py info

# Options avancées
python main.py embed \
    --input      data/processed/images \
    --output     data/processed/embeddings \
    --checkpoint models/embedding_model.pt \
    --format     hdf5 \
    --batch-size 32
```

Depuis Python :

```python
from src.process.embedding_model import FoodEmbeddingModel

model = FoodEmbeddingModel.from_config()
embeddings = model.encode(images_tensor, return_numpy=True)
# shape → (N, 256), dtype float32, norme L2 = 1
```

---

## Architecture du projet

```
nutri-ia-data-collection/
├── data/
│   ├── raw/         ← images et fichiers bruts (immuables)
│   ├── interim/     ← données en cours de traitement
│   └── processed/
│       ├── images/     ← images nettoyées, prêtes à l'embedding
│       ├── tabular/    ← CSV/JSON nettoyés
│       └── embeddings/ ← vecteurs générés (embeddings.h5)
├── models/          ← checkpoints sauvegardés (.pt)
├── src/
│   ├── config/
│   │   └── settings.py        ← configuration centralisée
│   ├── utils/
│   │   └── image_utils.py     ← dataset PyTorch + transformations
│   └── process/
│       ├── embedding_model.py ← le modèle
│       └── embed_images.py    ← pipeline de génération
├── tests/
│   └── test_embedding_model.py
├── docs/
│   └── EMBEDDING_MODEL.md     ← référence technique complète
├── main.py                    ← CLI
└── pyproject.toml
```

Les fichiers s'appellent dans cet ordre :

```
main.py
  └─ embed_images.py     (pipeline)
       ├─ embedding_model.py  (le modèle)
       ├─ image_utils.py      (le dataset)
       └─ settings.py         (la config)
```

---

## Comprendre le code

### Pourquoi un modèle d'embedding ?

Le projet collecte des photos d'aliments. Une image brute (grille de pixels) n'est pas exploitable directement par un assistant IA. Un **modèle d'embedding** transforme chaque image en un vecteur de nombres — une empreinte numérique qui capture ce que représente l'image.

Deux photos de pizza → vecteurs proches. Une pizza et une salade → vecteurs éloignés.

Ces vecteurs servent ensuite à :
- **Rechercher** des aliments visuellement similaires dans une base (FAISS, pgvector)
- **Alimenter** un modèle de classification nutritionnelle
- **Dédupliquer** le dataset (détecter les images quasi-identiques)

---

### 1. Configuration — `src/config/settings.py`

Centralise **tous les paramètres** du projet. Aucun "magic number" dispersé dans le code.

#### `Paths` — les chemins

```python
class Paths:
    raw_images:       Path = _ROOT / "data" / "raw" / "images"
    processed_images: Path = _ROOT / "data" / "processed" / "images"
    embeddings:       Path = _ROOT / "data" / "processed" / "embeddings"
    models:           Path = _ROOT / "models"
```

La méthode `ensure_all()` crée tous les dossiers manquants d'un seul appel.

#### `EmbeddingModelSettings` — les hyperparamètres

```python
class EmbeddingModelSettings(BaseSettings):
    backbone:              str   = "efficientnet_b2"
    pretrained:            bool  = True
    embedding_dim:         int   = 256
    projection_hidden_dim: int   = 512
    projection_dropout:    float = 0.2
    normalize_output:      bool  = True
    image_size:            int   = 224
    batch_size:            int   = 64
    device:                str   = "auto"   # auto → CUDA > MPS > CPU
    output_format:         str   = "hdf5"
```

Hérite de **Pydantic `BaseSettings`** : chaque paramètre est surchargeable sans toucher au code, via variable d'environnement ou fichier `.env` :

```bash
export NUTRIA_EMBED_BACKBONE=resnet50
export NUTRIA_EMBED_EMBEDDING_DIM=512
```

Pydantic valide aussi les types et les bornes au démarrage — une valeur invalide lève une erreur immédiatement, pas au milieu d'un entraînement.

```python
# Deux instances globales importées par tous les autres modules
paths     = Paths()
embed_cfg = EmbeddingModelSettings()
```

---

### 2. Préparation des images — `src/utils/image_utils.py`

#### Les transformations

**Inférence** (`get_inference_transforms`) — pipeline déterministe :

```python
transforms.Resize(256)        # redimensionne avec marge
transforms.CenterCrop(224)    # crop centré → 224×224
transforms.ToTensor()         # pixels [0,255] → tenseur [0.0, 1.0]
transforms.Normalize(         # normalisation ImageNet
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225]
)
```

La même image donne toujours le même tenseur → embeddings reproductibles. La normalisation ImageNet est obligatoire : le backbone a été pré-entraîné avec ces valeurs.

**Entraînement** (`get_train_transforms`) — pipeline avec augmentation :

```python
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))  # zoom aléatoire
transforms.RandomHorizontalFlip(p=0.5)               # miroir 50%
transforms.ColorJitter(brightness=0.2, ...)          # variations couleur
transforms.RandomGrayscale(p=0.05)                   # niveaux de gris 5%
```

Chaque passage crée une version légèrement différente de l'image → le modèle voit artificiellement plus de diversité → meilleure généralisation.

#### `FoodImageDataset`

Hérite de `torch.utils.data.Dataset`. PyTorch le passe à un `DataLoader` pour charger les images en parallèle par batches.

**Détection automatique des classes** : si le dossier contient des sous-dossiers, ils deviennent les classes :

```
images/
├── pizza/    → label 0
├── salade/   → label 1
└── burger/   → label 2
```

Structure plate (pas de sous-dossiers) → images chargées sans label.

**Résilience** : une image corrompue est remplacée par une image noire avec un avertissement — le batch n'est jamais cassé.

---

### 3. Le modèle — `src/process/embedding_model.py`

C'est le cœur du projet. Deux classes : `ProjectionHead` et `FoodEmbeddingModel`.

#### Architecture

```
Image RGB  (B × 3 × 224 × 224)
      │
  Backbone (timm)               ← réseau pré-entraîné ImageNet
      │  (B × 1408)             ← features brutes EfficientNet-B2
      │
  Projection Head (MLP)         ← couches denses entraînables
      │  (B × 256)
      │
  L2 Normalize                  ← ‖embedding‖₂ = 1
      │
  Embedding final (B × 256)
```

`B` = taille du batch (64 images traitées en parallèle).

#### `ProjectionHead` — la tête de projection

```python
nn.Linear(1408, 512, bias=False)   # couche 1 : réduction
nn.BatchNorm1d(512)
nn.ReLU(inplace=True)
nn.Dropout(p=0.2)

nn.Linear(512, 256, bias=False)    # couche 2 : compression
nn.BatchNorm1d(256)
nn.ReLU(inplace=True)
nn.Dropout(p=0.2)

nn.Linear(256, 256, bias=True)     # sortie
```

**Pourquoi une tête de projection ?** Le backbone sort des features optimisées pour 1000 classes ImageNet génériques. La projection est la partie **entraînable** qui adapte ces features au domaine alimentaire, sans ré-entraîner les 9M de paramètres du backbone.

Détail des composants :
- `Linear` — transformation matricielle apprise
- `BatchNorm1d` — normalise les activations → stabilise l'entraînement
- `ReLU` — activation non-linéaire (met les négatifs à zéro)
- `Dropout(0.2)` — éteint 20% des neurones aléatoirement → évite le surapprentissage
- Initialisation **He (Kaiming)** — calibrée pour ReLU → convergence plus rapide

#### `FoodEmbeddingModel` — le modèle principal

**Chargement du backbone via timm :**

```python
self.backbone = timm.create_model(
    "efficientnet_b2",
    pretrained=True,   # poids ImageNet téléchargés
    num_classes=0,     # supprime la tête de classification
    global_pool="avg", # Global Average Pooling → vecteur 1D
)
```

`timm` donne accès à +700 architectures avec une ligne. `num_classes=0` retire la dernière couche : on veut les features intermédiaires, pas une prédiction de classe.

**Gel du backbone :**

```python
# Extraction pure : seule la projection est entraînée (~10× plus rapide)
model = FoodEmbeddingModel.from_config(freeze_backbone=True)

# Fine-tuning partiel : adapter les 2 derniers blocs au domaine alimentaire
model = FoodEmbeddingModel.from_config(
    freeze_backbone=True,
    unfreeze_last_n_blocks=2
)
```

`requires_grad=False` sur le backbone → PyTorch ne calcule pas les gradients → ~10× plus rapide, 90% moins de mémoire GPU.

**Les trois méthodes d'utilisation :**

```python
# 1. forward() — entraînement (avec gradient)
embeddings = model(images)                              # (B, 256)

# 2. encode() — inférence sans gradient
embeddings = model.encode(images, return_numpy=True)   # np.ndarray float32

# 3. extract_features() — features brutes avant projection
features = model.extract_features(images)              # (B, 1408)
```

**Normalisation L2 :**

```python
embeddings = F.normalize(embeddings, p=2, dim=1)
# Chaque vecteur divisé par sa norme → ‖e‖₂ = 1
```

Après normalisation, la **similarité cosinus** entre deux embeddings est leur simple **produit scalaire** — opération O(D) ultrapide, compatible FAISS et pgvector.

**Sérialisation :**

```python
model.save("models/embedding_model.pt")               # sauvegarde poids + métadonnées
model = FoodEmbeddingModel.load("models/embedding_model.pt")  # reconstruction exacte
```

Le checkpoint `.pt` contient les poids et les métadonnées (backbone, dimension, normalisation). `load()` reconstruit l'architecture exacte automatiquement.

---

### 4. Le pipeline — `src/process/embed_images.py`

Orchestre tout : dossier d'images en entrée → embeddings sauvegardés en sortie.

#### `EmbeddingGenerator`

```python
generator = EmbeddingGenerator(model=model, device=device, batch_size=64)
data = generator.generate(dataset)
# {"embeddings": (N, 256), "paths": (N,), "labels": (N,)}
```

`generate()` :
1. Crée un `DataLoader` — chargement parallèle des images
2. Boucle sur les batches avec une barre de progression (`tqdm`)
3. Appelle `model.encode()` sur chaque batch
4. Accumule les résultats, mesure la vitesse en img/s
5. Retourne un dictionnaire de tableaux NumPy

`pin_memory=True` sur CUDA : tenseurs en mémoire épinglée → transfert CPU→GPU plus rapide.

#### Les formats de sauvegarde

| Format | Fichier produit | Avantage |
|--------|-----------------|----------|
| `hdf5` (**défaut**) | `embeddings.h5` | Compressé (gzip 4), métadonnées, chargement partiel |
| `npy` | `embeddings.npy` + `paths.npy` | Simple, rapide à charger |
| `parquet` | `embeddings.parquet` | Compatible pandas / SQL |

#### `run_embedding_pipeline()`

```python
run_embedding_pipeline(
    input_dir="data/processed/images",
    output_dir="data/processed/embeddings",
    checkpoint="models/embedding_model.pt",  # optionnel
    output_format="hdf5",
    batch_size=64,
)
```

Enchaîne chargement modèle → dataset → génération → sauvegarde. C'est la fonction appelée par `main.py`.

---

### 5. Point d'entrée — `main.py`

Deux commandes disponibles :

```bash
python main.py embed   # génère les embeddings
python main.py info    # affiche la config et les stats du modèle
```

`python main.py info` affiche :

```
=== Configuration active ===
  backbone                      : efficientnet_b2
  embedding_dim                 : 256
  device                        : auto
  output_format                 : hdf5
  ...

=== Résumé du modèle ===
FoodEmbeddingModel(
  backbone=efficientnet_b2,
  embedding_dim=256,
  params_total=10,345,984,
  params_trainable=10,345,984
)
```

---

## Flux complet d'une exécution

```
python main.py embed
        │
        ▼
cmd_embed()                                   [main.py]
        │
        ▼
run_embedding_pipeline()                      [embed_images.py]
        │
        ├─ FoodEmbeddingModel.from_config()   [embedding_model.py]
        │       └─ timm.create_model("efficientnet_b2", pretrained=True)
        │
        ├─ FoodImageDataset(root_dir)         [image_utils.py]
        │       └─ détecte N images dans K classes
        │
        ├─ EmbeddingGenerator.generate()      [embed_images.py]
        │       └─ batches × 64 images
        │               └─ model.encode(batch) → np.ndarray(64, 256)
        │
        └─ EmbeddingGenerator.save()          [embed_images.py]
                └─ data/processed/embeddings/embeddings.h5
                        ├─ embeddings  (N, 256)  float32  compressé
                        ├─ paths       (N,)      string
                        └─ labels      (N,)      int32
```

---

## Choix technologiques

| Choix | Raison |
|-------|--------|
| **timm** | +700 backbones, API unifiée, poids pré-entraînés intégrés |
| **Pydantic Settings** | Config validée, surchargeable par variable d'env sans modifier le code |
| **HDF5 (h5py)** | Compression native, chargement partiel, format standard ML |
| **`@torch.no_grad()`** | Désactive les gradients en inférence → 2× plus rapide, moitié moins de mémoire |
| **Normalisation L2** | Produit scalaire = similarité cosinus → compatible FAISS / pgvector |
| **BatchNorm dans la projection** | Stabilise l'entraînement, réduit la sensibilité au learning rate |
| **Initialisation He (Kaiming)** | Calibrée pour ReLU → évite la disparition des gradients |

---

## Tests

```bash
uv run pytest tests/ -v
```

28 tests couvrant : instanciation, shapes des sorties, normalisation, gel/dégel, sérialisation, dataset, transformations, et les trois formats de sauvegarde.

La référence technique complète (API, formats de sortie, tableaux de backbones) est dans [`EMBEDDING_MODEL.md`](EMBEDDING_MODEL.md).
