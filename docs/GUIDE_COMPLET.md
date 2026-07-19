# Guide complet — Nutri-IA : Code, Notions Techniques et Tests

> Ce document explique le projet de bout en bout : ce que fait chaque fichier,
> les concepts techniques utilisés, et comment tester et exécuter le code.

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Installation et exécution](#2-installation-et-exécution)
3. [Architecture des fichiers](#3-architecture-des-fichiers)
4. [Notions techniques fondamentales](#4-notions-techniques-fondamentales)
5. [Explication détaillée du code](#5-explication-détaillée-du-code)
   - [settings.py — Configuration](#51-settingspy--configuration)
   - [image_utils.py — Gestion des images](#52-image_utilspy--gestion-des-images)
   - [prepare_images.py — Prétraitement raw → processed](#53-prepare_imagespy--prétraitement-raw--processed)
   - [embedding_model.py — Le modèle](#54-embedding_modelpy--le-modèle)
   - [embed_images.py — Le pipeline d'embedding](#55-embed_imagespy--le-pipeline-dembedding)
   - [main.py — Le point d'entrée CLI](#56-mainpy--le-point-dentrée-cli)
6. [Workflow complet — de raw à embeddings](#6-workflow-complet--de-raw-à-embeddings)
7. [Flux d'exécution détaillé](#7-flux-dexécution-détaillé)
8. [Les tests — Explication et exécution](#8-les-tests--explication-et-exécution)

---

## 1. Vue d'ensemble du projet

**Nutri-IA** est un pipeline de traitement d'images alimentaires.

**Objectif** : transformer des photos de nourriture en vecteurs numériques (appelés
**embeddings**) qu'un système IA peut exploiter pour rechercher des aliments similaires,
les classifier, ou les dédupliquer.

```
Photo de pizza → [0.12, -0.45, 0.87, ..., 0.03]  ← vecteur de 256 nombres
Photo de salade → [−0.88, 0.21, −0.34, ..., 0.71] ← vecteur différent
```

Deux photos de la même catégorie → vecteurs proches.
Des aliments différents → vecteurs éloignés.

Ces vecteurs sont ensuite stockés dans une base de données vectorielle (FAISS, pgvector)
pour faire de la recherche par similarité en quelques millisecondes.


---

## 2. Installation et exécution

### Prérequis

- Python 3.10 ou plus récent
- `uv` (gestionnaire de paquets moderne, plus rapide que pip)

### Étapes

```bash
# 1. Aller dans le dossier du projet
cd /Users/tiesko/Documents/DAH/nutri

# 2. Installer uv si pas encore fait
pip install uv

# 3. Installer toutes les dépendances (dont les outils de dev/test)
uv sync --extra dev

# 4. Vérifier que tout fonctionne (affiche la config et les stats du modèle)
uv run python main.py info

# 5. Lancer les tests
uv run pytest tests/ -v
```

### Commandes disponibles

| Commande | Description |
|---|---|
| `uv run python main.py prepare` | Prétraite les images raw → processed (étape 1) |
| `uv run python main.py embed` | Génère les embeddings (étape 2) |
| `uv run python main.py info` | Affiche la configuration active et les stats du modèle |
| `uv run pytest tests/ -v` | Lance tous les tests unitaires |
| `uv run pytest tests/ --cov=src --cov-report=term-missing` | Tests + couverture de code |

### Workflow typique complet

```bash
# Étape 1 — Prétraitement : raw → processed
# Convertit en RGB, nettoie les fichiers parasites, sauvegarde en JPEG
uv run python main.py prepare

# Étape 2 — Embeddings : processed → vecteurs
# Génère les embeddings et les sauvegarde dans data/processed/embeddings/
uv run python main.py embed
```

### Options de la commande prepare

```bash
uv run python main.py prepare \
    --input    data/raw/images \      # source (défaut)
    --output   data/processed/images \  # destination (défaut)
    --max-size 1024 \                 # redimensionne si > 1024px (optionnel)
    --dry-run \                       # simule sans écrire
    --overwrite                       # réécrit les fichiers déjà existants
```

### Options du pipeline embed

```bash
uv run python main.py embed \
    --input      data/processed/images \   # dossier d'images source
    --output     data/processed/embeddings \  # dossier de sortie
    --checkpoint models/embedding_model.pt \  # poids existants (optionnel)
    --format     hdf5 \                    # hdf5, npy ou parquet
    --batch-size 32                        # images traitées en parallèle
```

### Surcharger la configuration sans toucher au code

```bash
# Utiliser ResNet-50 au lieu d'EfficientNet-B2
export NUTRIA_EMBED_BACKBONE=resnet50

# Réduire la dimension d'embedding
export NUTRIA_EMBED_EMBEDDING_DIM=128

# Forcer le CPU même si un GPU est disponible
export NUTRIA_EMBED_DEVICE=cpu
```


---

## 3. Architecture des fichiers

```
nutri/
├── main.py                          ← point d'entrée CLI (commandes prepare / embed / info)
├── pyproject.toml                   ← dépendances et config du projet
│
├── src/
│   ├── config/
│   │   └── settings.py              ← TOUS les paramètres centralisés ici
│   ├── utils/
│   │   └── image_utils.py           ← chargement images + transformations
│   └── process/
│       ├── prepare_images.py        ← prétraitement raw → processed  ← NOUVEAU
│       ├── embedding_model.py       ← le modèle de deep learning (cœur du projet)
│       └── embed_images.py          ← pipeline : images → fichier d'embeddings
│
├── tests/
│   └── test_embedding_model.py      ← 28 tests unitaires
│
├── data/
│   ├── raw/
│   │   └── images/                  ← images brutes ORIGINALES (ne pas modifier)
│   │       ├── alloco/              ← 100 images (IMG_CIV03_001.jpg … 100)
│   │       └── mafe/                ← (à compléter)
│   ├── interim/images/              ← images en cours de traitement (optionnel)
│   └── processed/
│       ├── images/                  ← images nettoyées, prêtes à l'embedding
│       │   ├── alloco/              ← copies converties en JPEG RGB
│       │   └── mafe/
│       └── embeddings/              ← vecteurs générés (embeddings.h5)
│
└── models/                          ← checkpoints sauvegardés (.pt)
```

**Règle fondamentale :** `data/raw/` est **en lecture seule**. On ne modifie jamais
les fichiers bruts. Toutes les opérations écrivent dans `processed/`.

**Ordre d'appel des fichiers :**

```
main.py
  ├── prepare_images.py  (raw → processed)
  └── embed_images.py    (pipeline d'embedding)
        ├── embedding_model.py  (le modèle)
        ├── image_utils.py      (le dataset)
        └── settings.py         (la config)
```


---

## 4. Notions techniques fondamentales

Cette section explique les concepts clés utilisés dans le projet,
sans supposer de connaissances préalables en deep learning.

### 4.1 Qu'est-ce qu'un embedding ?

Un **embedding** est une représentation numérique d'une donnée (ici, une image)
sous forme d'un vecteur de nombres réels.

```
Image (224×224 pixels × 3 couleurs = 150 528 valeurs brutes)
        ↓  modèle de deep learning
Embedding (256 nombres)  ← représentation compressée et sémantique
```

La compression n'est pas aléatoire : le modèle apprend à mettre des images
**similaires** proches dans cet espace, et des images **différentes** éloignées.

### 4.2 Backbone pré-entraîné (Transfer Learning)

Entraîner un réseau de neurones depuis zéro sur des images de nourriture
nécessiterait des millions d'images et des semaines de calcul GPU.

Le **Transfer Learning** (apprentissage par transfert) contourne ce problème :
on part d'un réseau déjà entraîné sur **ImageNet** (1,2 million d'images, 1000 classes),
qui sait déjà reconnaître des textures, formes, contours, et objets génériques.
On lui ajoute seulement une petite tête de projection entraînable.

```
EfficientNet-B2 (pré-entraîné ImageNet) → résultats utiles dès le départ
        + Projection Head (entraînable)  → adaptation au domaine alimentaire
```

Ce projet utilise **timm** (PyTorch Image Models), une bibliothèque qui donne
accès à plus de 700 architectures de backbones en une ligne de code.

### 4.3 Normalisation L2

Après la projection, chaque vecteur est divisé par sa propre norme :

```
e_normalisé = e / ‖e‖₂    →    ‖e_normalisé‖₂ = 1
```

Tous les vecteurs se retrouvent sur une **sphère unité**. Cela a un avantage
pratique majeur : la **similarité cosinus** entre deux vecteurs normalisés
se réduit à un simple produit scalaire — opération ultrarapide dans FAISS.

```
similarité(a, b) = a · b    (si ‖a‖ = ‖b‖ = 1)
```

### 4.4 Gel du backbone (Freeze)

Le backbone contient des millions de paramètres. En mode **extraction de features**
(pas d'entraînement), on n'a pas besoin de calculer leurs gradients.

```python
freeze_backbone=True  # requires_grad=False sur tous les params du backbone
```

Effet : **~10× plus rapide**, **~50% moins de mémoire GPU**.

Le backbone gelé agit comme un extracteur de features fixe.
Seule la tête de projection apprend (quelques milliers de paramètres).

### 4.5 BatchNorm, ReLU, Dropout

Ces trois composants apparaissent dans la tête de projection :

**BatchNorm1d** : normalise les activations de chaque couche pour éviter
que les valeurs explosent ou s'effondrent pendant l'entraînement.
Stabilise et accélère la convergence.

**ReLU** (Rectified Linear Unit) : `f(x) = max(0, x)`.
Introduit la non-linéarité — sans elle, empiler des couches linéaires
n'aurait aucun intérêt (le résultat resterait linéaire).

**Dropout(p=0.2)** : éteint aléatoirement 20% des neurones à chaque passage.
Force le réseau à ne pas dépendre d'un seul neurone → meilleure généralisation.
Désactivé automatiquement en mode évaluation (`model.eval()`).

### 4.6 Pydantic Settings

Pydantic est une bibliothèque de validation de données Python.
`BaseSettings` étend Pydantic pour lire des variables d'environnement
et fichiers `.env`, en plus de la validation de types et bornes.

```python
embedding_dim: int = Field(default=256, ge=32, le=2048)
# ge=32 → doit être >= 32, le=2048 → doit être <= 2048
# Une valeur invalide → erreur immédiate au démarrage, pas en pleine exécution
```

### 4.7 Dataset et DataLoader PyTorch

`Dataset` : interface PyTorch pour accéder aux données (méthodes `__len__` et `__getitem__`).

`DataLoader` : enveloppe un Dataset pour charger les données par **batches**,
en **parallèle** (plusieurs workers), avec des **shuffles** optionnels.

```
FoodImageDataset[0]  → {"image": tensor(3,224,224), "path": "...", "label": 1}
FoodImageDataset[1]  → ...
        ↓ DataLoader (batch_size=64, num_workers=4)
Batch  →  {"image": tensor(64,3,224,224), "path": [...], "label": tensor(64)}
```

Le DataLoader alimente le modèle en continu pendant qu'il traite le batch précédent.


---

## 5. Explication détaillée du code

### 5.1 `settings.py` — Configuration

**Rôle** : centraliser TOUS les paramètres. Aucun "magic number" éparpillé dans le code.

#### Classe `Paths`

```python
class Paths:
    root:             Path = _ROOT                          # racine du projet
    raw_images:       Path = _ROOT / "data/raw/images"      # images brutes
    processed_images: Path = _ROOT / "data/processed/images" # images propres
    embeddings:       Path = _ROOT / "data/processed/embeddings" # vecteurs
    models:           Path = _ROOT / "models"               # checkpoints .pt
```

`_ROOT` est calculé dynamiquement depuis l'emplacement du fichier — le projet
fonctionne quel que soit l'endroit où il est cloné.

`ensure_all()` crée tous les dossiers d'un seul appel, avec `parents=True`
(crée les dossiers intermédiaires si nécessaire).

#### Classe `EmbeddingModelSettings`

```python
class EmbeddingModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NUTRIA_EMBED_",  # préfixe des variables d'environnement
        env_file=".env",             # lecture automatique du fichier .env
    )

    backbone:              str   = "efficientnet_b2"  # architecture timm
    embedding_dim:         int   = 256                # taille du vecteur final
    projection_hidden_dim: int   = 512                # taille couches internes
    projection_dropout:    float = 0.2                # taux de dropout
    normalize_output:      bool  = True               # normalisation L2
    image_size:            int   = 224                # taille image (pixels)
    batch_size:            int   = 64                 # images par batch
    device:                str   = "auto"             # cpu / cuda / mps / auto
    output_format:         str   = "hdf5"             # format de sauvegarde
```

**Instances globales** importées par tous les autres modules :

```python
paths     = Paths()
embed_cfg = EmbeddingModelSettings()
```

Ainsi, `from src.config.settings import embed_cfg` donne accès à
`embed_cfg.backbone`, `embed_cfg.batch_size`, etc. partout dans le code.

---

### 5.2 `image_utils.py` — Gestion des images

**Rôle** : charger les images depuis le disque et les préparer pour le modèle.

#### Les transformations

Les images doivent être converties en tenseurs PyTorch normalisés avant
d'entrer dans le modèle. Deux pipelines existent :

**Inférence** (déterministe — même image = même résultat à chaque fois) :

```python
transforms.Resize(256)           # redimensionne (hauteur ou largeur = 256)
transforms.CenterCrop(224)       # crop centré → exactement 224×224
transforms.ToTensor()            # pixels [0-255] → tenseur float [0.0-1.0]
transforms.Normalize(            # normalisation ImageNet obligatoire :
    mean=[0.485, 0.456, 0.406],  # le backbone a été pré-entraîné avec ces valeurs
    std= [0.229, 0.224, 0.225]   # appliquer les mêmes stats = performances optimales
)
```

**Entraînement** (aléatoire — chaque passage crée une version différente) :

```python
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))  # zoom aléatoire 70-100%
transforms.RandomHorizontalFlip(p=0.5)               # miroir horizontal 50%
transforms.ColorJitter(brightness=0.2, ...)          # légère variation de couleur
transforms.RandomGrayscale(p=0.05)                   # niveaux de gris 5% du temps
```

L'augmentation augmente artificiellement la diversité du dataset vu par le modèle
→ meilleure robustesse (le modèle ne "mémorise" pas les images).

#### Classe `FoodImageDataset`

Hérite de `torch.utils.data.Dataset`. Deux comportements selon la structure :

**Structure hiérarchique** (recommandée pour un dataset avec labels) :

```
data/processed/images/
├── pizza/         → label 0
│   ├── img_001.jpg
│   └── img_002.jpg
├── salade/        → label 1
└── burger/        → label 2
```

**Structure plate** (pour générer des embeddings sans labels) :

```
data/processed/images/
├── img_001.jpg
├── img_002.jpg
└── img_003.jpg
```

**Résilience aux images corrompues** : au lieu de crasher le batch entier,
une image illisible est remplacée par une image noire et un avertissement est loggué.

```python
# Dans __getitem__ :
try:
    image = Image.open(path).convert("RGB")
except (UnidentifiedImageError, OSError):
    image = Image.new("RGB", (224, 224))   # image noire de remplacement
```


---

### 5.3 `prepare_images.py` — Prétraitement raw → processed

**Rôle** : copier et nettoyer les images brutes avant de les passer au modèle.
Les données `raw/` restent intactes. Toutes les copies propres vont dans `processed/`.

#### Ce que fait le script sur chaque image

```
data/raw/images/alloco/IMG_CIV03_001.jpg
        │
        │  1. Ouverture et vérification (image corrompue ? → ignorée)
        │  2. Conversion en RGB (supprime canal alpha RGBA, convertit grayscale L)
        │  3. Redimensionnement optionnel (si --max-size, préserve le ratio)
        │  4. Sauvegarde en JPEG qualité 95 (uniformise .png/.jpeg/.bmp → .jpg)
        │
        ▼
data/processed/images/alloco/IMG_CIV03_001.jpg
```

#### Comportement idempotent (safe à relancer)

Par défaut (`--overwrite` absent), le script **ignore** les images déjà présentes
dans `processed/`. Relancer la commande plusieurs fois ne duplique rien.

```bash
# Première exécution : traite 100 images
uv run python main.py prepare
# → 100 traitées, 0 ignorées, 0 erreurs

# Deuxième exécution : tout est déjà là
uv run python main.py prepare
# → 0 traitées, 100 ignorées, 0 erreurs

# Forcer le re-traitement (ex. après changement de --max-size)
uv run python main.py prepare --overwrite
```

#### Dry run — simuler sans écrire

Utile pour vérifier ce qui va être traité avant de lancer :

```bash
uv run python main.py prepare --dry-run
# Affiche : WOULD WRITE : data/processed/images/alloco/IMG_CIV03_001.jpg
#           WOULD WRITE : data/processed/images/alloco/IMG_CIV03_002.jpg
#           ...
```

#### Limiter la taille des images

Sur de gros datasets, des images très larges (4000×3000px) ralentissent l'inférence
sans apporter d'information utile (le modèle les redimensionne de toute façon à 224px).
`--max-size` redimensionne en amont pour gagner en espace disque et en vitesse de chargement.

```bash
# Images redimensionnées à 1024px max (ratio préservé)
uv run python main.py prepare --max-size 1024
```

#### Cas concret avec tes données

```bash
# Vérifier d'abord sans écrire
uv run python main.py prepare --dry-run

# Lancer le prétraitement
uv run python main.py prepare

# Résultat attendu :
# Classes détectées (2) : alloco, mafe
# 100 image(s) trouvée(s) dans data/raw/images
# Terminé — 100 traitées, 0 ignorées, 0 erreurs
```

#### Fichiers ignorés automatiquement

Le script filtre les fichiers parasites courants :
`.DS_Store`, `Thumbs.db`, `.gitkeep`, `desktop.ini` — aucun besoin de les supprimer manuellement.

---

### 5.4 `embedding_model.py` — Le modèle

**Rôle** : transformer un batch d'images en vecteurs d'embedding.

#### Fonction `resolve_device()`

```python
def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():   # GPU NVIDIA
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # GPU Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")      # fallback CPU
    return torch.device(preference)
```

Détecte automatiquement le meilleur accélérateur disponible.
Sur Mac M1/M2/M3, utilise MPS (Metal Performance Shaders) — ~3-5× plus rapide que CPU.

#### Classe `ProjectionHead`

Un MLP (Multi-Layer Perceptron) à 3 couches qui compresse les features du backbone
vers la dimension d'embedding finale :

```
in_features (ex. 1408 pour EfficientNet-B2)
    │
    ├── Linear(1408 → 512)   ← réduction
    ├── BatchNorm1d(512)      ← stabilisation
    ├── ReLU                  ← non-linéarité
    ├── Dropout(0.2)          ← régularisation
    │
    ├── Linear(512 → 256)    ← compression
    ├── BatchNorm1d(256)
    ├── ReLU
    ├── Dropout(0.2)
    │
    └── Linear(256 → 256)    ← sortie finale (embedding_dim)
```

**Initialisation He (Kaiming)** : les poids sont initialisés avec une distribution
calibrée pour ReLU, ce qui évite que les gradients s'effondrent au début de
l'entraînement.

#### Classe `FoodEmbeddingModel`

**Instanciation du backbone via timm :**

```python
self.backbone = timm.create_model(
    "efficientnet_b2",
    pretrained=True,   # télécharge les poids ImageNet (~30 Mo)
    num_classes=0,     # retire la tête de classification (on veut les features)
    global_pool="avg", # Global Average Pooling : feature map → vecteur 1D
)
# backbone_out_dim = 1408 pour EfficientNet-B2
```

**Global Average Pooling** : prend la moyenne spatiale de chaque feature map.
Transforme (B, 1408, 7, 7) en (B, 1408). Permet d'accepter des images de toute taille.

**Gel du backbone :**

```python
# Mode extraction pure (rapide, peu de mémoire)
model = FoodEmbeddingModel.from_config(freeze_backbone=True)

# Fine-tuning partiel : adapter les 2 derniers blocs
model = FoodEmbeddingModel.from_config(
    freeze_backbone=True,
    unfreeze_last_n_blocks=2
)
```

**Les 3 méthodes d'utilisation :**

```python
# 1. forward(x) — entraînement, avec calcul des gradients
embeddings = model(images)                # shape (B, 256)

# 2. encode(x) — inférence, @torch.no_grad() appliqué automatiquement
embeddings = model.encode(images, return_numpy=True)  # np.ndarray float32

# 3. extract_features(x) — features brutes avant la projection
features = model.extract_features(images)  # shape (B, 1408)
```

**Sérialisation (save/load) :**

Le checkpoint `.pt` contient les poids ET les métadonnées nécessaires à la
reconstruction du modèle. `load()` recrée l'architecture exacte automatiquement
sans que l'utilisateur ait à se souvenir des paramètres utilisés.

```python
# Sauvegarde
checkpoint = {
    "model_state_dict": self.state_dict(),   # tous les poids
    "backbone_name":    "efficientnet_b2",   # architecture
    "embedding_dim":    256,                 # dimension de sortie
    "normalize_output": True,                # flag de normalisation
}
torch.save(checkpoint, "models/embedding_model.pt")

# Chargement — reconstruction automatique
checkpoint = torch.load("models/embedding_model.pt")
model = FoodEmbeddingModel(
    backbone_name=checkpoint["backbone_name"],
    pretrained=False,                        # poids chargés depuis le fichier
    embedding_dim=checkpoint["embedding_dim"],
)
model.load_state_dict(checkpoint["model_state_dict"])
```


---

### 5.5 `embed_images.py` — Le pipeline d'embedding

**Rôle** : orchestrer la génération complète d'embeddings sur un dossier d'images.

#### Classe `EmbeddingGenerator`

```python
generator = EmbeddingGenerator(model=model, device=device, batch_size=64)
data = generator.generate(dataset)
# Retourne : {"embeddings": (N, 256), "paths": (N,), "labels": (N,)}
```

**Ce que fait `generate()` en interne :**

```python
loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

for batch in tqdm(loader):                          # barre de progression
    images = batch["image"].to(device)              # CPU → GPU
    embeddings = model.encode(images, return_numpy=True)  # inférence
    all_embeddings.append(embeddings)               # accumulation
    all_paths.extend(batch["path"])

# À la fin :
return {
    "embeddings": np.vstack(all_embeddings),  # (N, 256) float32
    "paths":      np.array(all_paths),        # (N,) strings
    "labels":     np.array(all_labels),       # (N,) int32  (si présents)
}
```

`pin_memory=True` : alloue les tenseurs en mémoire paginée fixe (non-swappable),
ce qui accélère le transfert CPU → GPU pour CUDA.

#### Les formats de sauvegarde

| Format | Fichier | Quand l'utiliser |
|--------|---------|-----------------|
| `hdf5` (défaut) | `embeddings.h5` | Production : compressé (gzip), métadonnées intégrées, chargement partiel possible |
| `npy` | `embeddings.npy` + `paths.npy` | Prototypage rapide : simple, chargement `np.load()` direct |
| `parquet` | `embeddings.parquet` | Analyse de données : compatible pandas, filtrage SQL |

**Lire les embeddings après génération :**

```python
# Format HDF5
import h5py
with h5py.File("data/processed/embeddings/embeddings.h5", "r") as f:
    embeddings = f["embeddings"][:]   # np.ndarray (N, 256)
    paths      = f["paths"][:]        # np.ndarray de strings

# Format NumPy
embeddings = np.load("data/processed/embeddings/embeddings.npy")
paths      = np.load("data/processed/embeddings/paths.npy", allow_pickle=True)

# Format Parquet
import pandas as pd
df = pd.read_parquet("data/processed/embeddings/embeddings.parquet")
```

#### Fonction `run_embedding_pipeline()`

Fonction all-in-one qui enchaîne toutes les étapes :

```python
def run_embedding_pipeline(input_dir, output_dir, checkpoint, output_format, batch_size):
    # 1. Charger le modèle (depuis checkpoint ou depuis la config)
    model = FoodEmbeddingModel.load(checkpoint)  # ou from_config()

    # 2. Créer le dataset
    dataset = FoodImageDataset(root_dir=input_dir, transform=get_inference_transforms())

    # 3. Générer les embeddings
    generator = EmbeddingGenerator(model=model, device=device)
    data = generator.generate(dataset)

    # 4. Sauvegarder
    EmbeddingGenerator.save(data, output_dir, fmt=output_format)
```

---

### 5.6 `main.py` — Le point d'entrée CLI

**Rôle** : exposer le pipeline via une interface en ligne de commande.

Utilise `argparse` (bibliothèque standard Python) pour parser les arguments.

Trois sous-commandes :

```bash
python main.py prepare  # prétraite les images raw → processed
python main.py embed    # génère les embeddings processed → .h5
python main.py info     # affiche la config et les stats du modèle
```

`python main.py info` affiche :

```
=== Configuration active ===
  backbone                      : efficientnet_b2
  pretrained                    : True
  embedding_dim                 : 256
  batch_size                    : 64
  device                        : auto
  output_format                 : hdf5
  ...

=== Résumé du modèle ===
  Paramètres total      :   10,345,984
  Paramètres entraîn.   :   10,345,984
  Paramètres gelés      :            0
```


---

## 6. Workflow complet — de raw à embeddings

Voici le chemin exact à suivre avec tes données (`alloco/`, `mafe/`).

### Étape 0 — Vérifier l'installation

```bash
uv run python main.py info
```

Doit afficher la config (backbone, embedding_dim, device…) sans erreur.
Si ça passe, l'environnement est prêt.

### Étape 1 — Prétraitement (raw → processed)

```bash
# Simulation d'abord pour voir ce qui va être traité
uv run python main.py prepare --dry-run

# Lancement réel
uv run python main.py prepare
```

**Ce qui se passe :**
- Scan de `data/raw/images/` → trouve 100 images dans `alloco/`
- Chaque image est vérifiée, convertie en RGB, sauvegardée en JPEG dans `data/processed/images/`
- La structure de classes est préservée : `processed/images/alloco/`

**Résultat attendu :**
```
Classes détectées (2) : alloco, mafe
100 image(s) trouvée(s) dans data/raw/images
Terminé — 100 traitées, 0 ignorées, 0 erreurs
```

### Étape 2 — Génération des embeddings (processed → vecteurs)

```bash
uv run python main.py embed
```

**Ce qui se passe :**
- Charge EfficientNet-B2 pré-entraîné (téléchargement ~30 Mo au premier lancement)
- Lit les 100 images depuis `data/processed/images/`
- Génère un vecteur de 256 dimensions par image
- Sauvegarde dans `data/processed/embeddings/embeddings.h5`

**Résultat attendu :**
```
Backbone 'efficientnet_b2' chargé — features: 1408d, pretrained=True
FoodImageDataset — 100 images trouvées (2 classes) dans data/processed/images
Génération embeddings: 100%|████| 2/2 [00:03<00:00]
100 embeddings générés en 3.2s (31 img/s) — dim=256
Embeddings sauvegardés (HDF5) → data/processed/embeddings/embeddings.h5
```

### Étape 3 — Vérifier les embeddings produits

```python
import h5py
import numpy as np

with h5py.File("data/processed/embeddings/embeddings.h5", "r") as f:
    emb = f["embeddings"][:]
    pth = f["paths"][:]

print(f"Shape    : {emb.shape}")                      # (100, 256)
print(f"Dtype    : {emb.dtype}")                      # float32
print(f"Norme[0] : {np.linalg.norm(emb[0]):.4f}")    # ≈ 1.0000
print(f"Chemin   : {pth[0]}")                         # .../alloco/IMG_CIV03_001.jpg
```

---

## 7. Flux d'exécution détaillé

```
python main.py prepare
        │
        ▼
  cmd_prepare()                                [main.py]
        │
        ▼
  run_prepare_pipeline()                       [prepare_images.py]
        │
        ├── scan data/raw/images/              → 100 fichiers image
        ├── filtre .DS_Store, .gitkeep…
        └── pour chaque image :
              Image.open() → convert("RGB")
              → resize si --max-size
              → save JPEG qualité 95
              → data/processed/images/alloco/IMG_CIV03_001.jpg

python main.py embed
        │
        ▼
  cmd_embed()                                  [main.py]
        │
        ▼
  run_embedding_pipeline()                     [embed_images.py]
        │
        ├──► FoodEmbeddingModel.from_config()  [embedding_model.py]
        │         │
        │         ├── timm.create_model("efficientnet_b2", pretrained=True)
        │         │       └── télécharge les poids ImageNet (~30 Mo, une seule fois)
        │         ├── ProjectionHead(1408 → 512 → 256)
        │         └── freeze_backbone() si demandé
        │
        ├──► FoodImageDataset(root_dir)         [image_utils.py]
        │         │
        │         ├── scanne le dossier → liste des images
        │         ├── détecte les classes (sous-dossiers)
        │         └── prépare les transformations d'inférence
        │
        ├──► EmbeddingGenerator.generate()      [embed_images.py]
        │         │
        │         └── DataLoader (batch_size=64, num_workers=4)
        │               └── pour chaque batch :
        │                     image.to(device)       # CPU → GPU
        │                     model.encode(image)    # inférence sans gradient
        │                     accumule embeddings + paths + labels
        │
        └──► EmbeddingGenerator.save()           [embed_images.py]
                  │
                  └── data/processed/embeddings/embeddings.h5
                            ├── embeddings  (N, 256)  float32  gzip
                            ├── paths       (N,)      string
                            └── labels      (N,)      int32
```

---

## 8. Les tests — Explication et exécution

### 8.1 Lancer les tests

```bash
# Tous les tests
uv run pytest tests/ -v

# Avec couverture de code
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Un seul groupe de tests
uv run pytest tests/ -v -k "TestForwardPasses"

# Un seul test précis
uv run pytest tests/ -v -k "test_forward_shape"
```

### 8.2 Pourquoi les tests utilisent ResNet-18

Tous les tests instancient le modèle avec `backbone_name="resnet18", pretrained=False`.

- **ResNet-18** est le backbone le plus léger de timm (~11M paramètres vs 30M+ pour EfficientNet-B2)
- `pretrained=False` évite de télécharger des poids (~30 Mo) à chaque test
- Les tests s'exécutent en quelques secondes, même sans GPU

```python
@pytest.fixture(scope="module")
def small_model() -> FoodEmbeddingModel:
    return FoodEmbeddingModel(
        backbone_name="resnet18",
        pretrained=False,
        embedding_dim=64,        # dimension réduite pour la rapidité
        projection_hidden_dim=128,
        projection_dropout=0.0,  # désactivé pour la reproductibilité
    )
```

`scope="module"` : le modèle est créé une seule fois pour tous les tests du module,
pas recréé à chaque test.

### 8.3 Explication de chaque groupe de tests

#### `test_resolve_device_*`

Vérifie que la détection du device fonctionne :

```python
def test_resolve_device_cpu():
    device = resolve_device("cpu")
    assert device == torch.device("cpu")

def test_resolve_device_auto_returns_device():
    device = resolve_device("auto")
    assert isinstance(device, torch.device)  # retourne toujours un device valide
```

#### `TestProjectionHead`

Vérifie que la tête de projection produit la bonne forme de sortie :

```python
def test_output_shape(self):
    head = ProjectionHead(in_features=512, hidden_dim=256, out_dim=64)
    x = torch.randn(8, 512)   # 8 vecteurs de 512 dimensions
    out = head(x)
    assert out.shape == (8, 64)   # doit sortir 8 vecteurs de 64 dimensions

def test_batch_size_one(self):
    # BatchNorm avec batch=1 en mode train est instable
    # En mode eval, ça doit fonctionner
    head.eval()
    out = head(torch.randn(1, 128))
    assert out.shape == (1, 32)
```

#### `TestFoodEmbeddingModelInit`

Vérifie l'instanciation et la détection d'erreurs :

```python
def test_from_config(self):
    cfg = EmbeddingModelSettings(backbone="resnet18", pretrained=False)
    model = FoodEmbeddingModel.from_config(cfg)
    assert model.embedding_dim == 32

def test_invalid_backbone_raises(self):
    # Un backbone inexistant doit lever ValueError immédiatement
    with pytest.raises(ValueError, match="non reconnu"):
        FoodEmbeddingModel(backbone_name="not_a_real_backbone_xyz")
```

#### `TestForwardPasses`

Vérifie les formes de sortie et la normalisation :

```python
def test_forward_shape(self, small_model, dummy_batch):
    # dummy_batch = torch.randn(4, 3, 224, 224)  → 4 images RGB 224×224
    out = small_model(dummy_batch)
    assert out.shape == (4, 64)   # 4 images → 4 embeddings de dim 64

def test_forward_normalized(self, small_model, dummy_batch):
    out = small_model(dummy_batch)
    norms = out.norm(dim=1)         # norme L2 de chaque vecteur
    # Tous les vecteurs doivent avoir une norme = 1 (±1e-5 pour les erreurs float)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

def test_encode_returns_numpy(self, small_model, dummy_batch):
    arr = small_model.encode(dummy_batch, return_numpy=True)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32   # float32 et non float64
```

#### `TestFreeze`

Vérifie que le gel/dégel fonctionne correctement :

```python
def test_freeze_backbone_no_grad(self):
    model = FoodEmbeddingModel(backbone_name="resnet18", freeze_backbone=True)
    for param in model.backbone.parameters():
        assert not param.requires_grad   # gelé → pas de gradient

def test_projection_head_trainable_when_frozen(self):
    model = FoodEmbeddingModel(backbone_name="resnet18", freeze_backbone=True)
    for param in model.projection.parameters():
        assert param.requires_grad   # la projection reste entraînable

def test_unfreeze_backbone(self):
    model = FoodEmbeddingModel(backbone_name="resnet18", freeze_backbone=True)
    model.unfreeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad   # dégelé → gradient réactivé
```

#### `TestSerialization`

Vérifie que save/load préserve exactement les poids :

```python
def test_save_and_load(self, small_model, tmp_path, dummy_batch):
    ckpt = tmp_path / "model.pt"
    small_model.save(ckpt)
    assert ckpt.exists()

    loaded = FoodEmbeddingModel.load(ckpt)

    # Les deux modèles doivent produire exactement les mêmes sorties
    with torch.no_grad():
        out_orig   = small_model(dummy_batch)
        out_loaded = loaded(dummy_batch)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)
```

`tmp_path` est une fixture pytest qui crée un dossier temporaire automatiquement
supprimé après le test.

#### `TestFoodImageDataset`

Utilise `tmp_image_dir`, une fixture qui crée 6 images PNG factices dans 2 classes :

```python
@pytest.fixture()
def tmp_image_dir(tmp_path):
    for cls in ("pizza", "salade"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            img = Image.new("RGB", (128, 128), color=(i*50, i*30, i*70))
            img.save(cls_dir / f"img_{i}.png")
    return tmp_path
```

```python
def test_len(self, tmp_image_dir):
    ds = FoodImageDataset(root_dir=tmp_image_dir)
    assert len(ds) == 6   # 2 classes × 3 images

def test_image_tensor_shape(self, tmp_image_dir):
    item = FoodImageDataset(root_dir=tmp_image_dir)[0]
    assert item["image"].shape == (3, 224, 224)   # RGB, 224×224

def test_classes_found(self, tmp_image_dir):
    ds = FoodImageDataset(root_dir=tmp_image_dir)
    assert set(ds.classes) == {"pizza", "salade"}
```

#### `TestTransforms`

Vérifie les transformations image :

```python
def test_inference_is_deterministic(self):
    img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    t = get_inference_transforms()
    # Deux appels sur la même image → même tenseur (pas d'aléatoire en inférence)
    assert torch.allclose(t(img), t(img))
```

#### `TestEmbeddingGeneratorSave`

Vérifie les 3 formats de sauvegarde avec des données factices :

```python
def test_save_hdf5(self, dummy_data, tmp_path):
    out = EmbeddingGenerator.save(dummy_data, tmp_path, fmt="hdf5")
    assert out.exists()
    with h5py.File(out, "r") as f:
        assert f["embeddings"].shape == (10, 64)
```

### 8.4 Résultats attendus

```
tests/test_embedding_model.py::test_resolve_device_cpu                  PASSED
tests/test_embedding_model.py::test_resolve_device_auto_returns_device  PASSED
tests/test_embedding_model.py::TestProjectionHead::test_output_shape    PASSED
... (28 tests au total)

========================= 28 passed in X.XXs =========================
```

### 8.5 Tester manuellement depuis Python

```python
import torch
from src.process.embedding_model import FoodEmbeddingModel

# Modèle léger pour test interactif
model = FoodEmbeddingModel(backbone_name="resnet18", pretrained=False, embedding_dim=64)
model.eval()

# Batch factice : 2 images RGB 224×224
x = torch.randn(2, 3, 224, 224)

# Vérifier la forme de sortie
out = model(x)
print(out.shape)   # torch.Size([2, 64])

# Vérifier la normalisation L2
print(out.norm(dim=1))   # tensor([1.0000, 1.0000]) — normes ≈ 1

# Tester encode() avec retour NumPy
arr = model.encode(x, return_numpy=True)
print(type(arr), arr.shape, arr.dtype)
# <class 'numpy.ndarray'> (2, 64) float32

# Compter les paramètres
print(model.count_parameters())
# {'trainable': ..., 'total': ..., 'frozen': 0}
```

---

*Document généré pour le projet Nutri-IA — Pipeline d'embedding d'images alimentaires.*
