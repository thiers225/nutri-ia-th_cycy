"""
Tests unitaires — Modèle d'embedding Nutri-IA.

Couvre :
- Instanciation et configuration du modèle
- Shape des sorties (forward, encode, extract_features)
- Normalisation L2
- Gel / dégel des paramètres
- Sérialisation (save / load)
- ProjectionHead
- FoodImageDataset (avec images temporaires)
- EmbeddingGenerator + sauvegarde multi-format
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config.settings import EmbeddingModelSettings
from src.process.embedding_model import FoodEmbeddingModel, ProjectionHead, resolve_device
from src.utils.image_utils import (
    FoodImageDataset,
    get_inference_transforms,
    get_train_transforms,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model() -> FoodEmbeddingModel:
    """Modèle léger (ResNet-18) sans poids pré-entraînés pour les tests."""
    return FoodEmbeddingModel(
        backbone_name="resnet18",
        pretrained=False,
        embedding_dim=64,
        projection_hidden_dim=128,
        projection_dropout=0.0,
        normalize_output=True,
    )


@pytest.fixture(scope="module")
def dummy_batch() -> torch.Tensor:
    """Batch de 4 images RGB 224×224."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture()
def tmp_image_dir(tmp_path: Path) -> Path:
    """Répertoire temporaire avec 6 images PNG factices dans 2 classes."""
    for cls in ("pizza", "salade"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            img = Image.new("RGB", (128, 128), color=(i * 50, i * 30, i * 70))
            img.save(cls_dir / f"img_{i}.png")
    return tmp_path


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------

def test_resolve_device_cpu() -> None:
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_resolve_device_auto_returns_device() -> None:
    device = resolve_device("auto")
    assert isinstance(device, torch.device)


# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------

class TestProjectionHead:
    def test_output_shape(self) -> None:
        head = ProjectionHead(in_features=512, hidden_dim=256, out_dim=64)
        x = torch.randn(8, 512)
        out = head(x)
        assert out.shape == (8, 64)

    def test_batch_size_one(self) -> None:
        """BN en mode train avec batch_size=1 doit lever une erreur ou être en eval."""
        head = ProjectionHead(in_features=128, hidden_dim=64, out_dim=32)
        head.eval()
        x = torch.randn(1, 128)
        out = head(x)
        assert out.shape == (1, 32)

    def test_weights_initialized(self) -> None:
        head = ProjectionHead(in_features=128, hidden_dim=64, out_dim=16)
        for m in head.modules():
            if isinstance(m, torch.nn.Linear):
                # He init — poids non nuls
                assert m.weight.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# FoodEmbeddingModel — instanciation
# ---------------------------------------------------------------------------

class TestFoodEmbeddingModelInit:
    def test_from_config(self) -> None:
        cfg = EmbeddingModelSettings(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=32,
        )
        model = FoodEmbeddingModel.from_config(cfg)
        assert model.embedding_dim == 32

    def test_invalid_backbone_raises(self) -> None:
        with pytest.raises(ValueError, match="non reconnu"):
            FoodEmbeddingModel(backbone_name="not_a_real_backbone_xyz", pretrained=False)

    def test_count_parameters(self, small_model: FoodEmbeddingModel) -> None:
        counts = small_model.count_parameters()
        assert counts["total"] > 0
        assert counts["trainable"] + counts["frozen"] == counts["total"]

    def test_repr_contains_backbone(self, small_model: FoodEmbeddingModel) -> None:
        assert "resnet18" in repr(small_model)


# ---------------------------------------------------------------------------
# Passes forward
# ---------------------------------------------------------------------------

class TestForwardPasses:
    def test_forward_shape(
        self, small_model: FoodEmbeddingModel, dummy_batch: torch.Tensor
    ) -> None:
        small_model.eval()
        out = small_model(dummy_batch)
        assert out.shape == (4, 64)

    def test_forward_normalized(
        self, small_model: FoodEmbeddingModel, dummy_batch: torch.Tensor
    ) -> None:
        small_model.eval()
        out = small_model(dummy_batch)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), "Embeddings non normalisés"

    def test_encode_returns_numpy(
        self, small_model: FoodEmbeddingModel, dummy_batch: torch.Tensor
    ) -> None:
        arr = small_model.encode(dummy_batch, return_numpy=True)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 64)
        assert arr.dtype == np.float32

    def test_encode_returns_tensor(
        self, small_model: FoodEmbeddingModel, dummy_batch: torch.Tensor
    ) -> None:
        t = small_model.encode(dummy_batch, return_numpy=False)
        assert isinstance(t, torch.Tensor)

    def test_extract_features_shape(
        self, small_model: FoodEmbeddingModel, dummy_batch: torch.Tensor
    ) -> None:
        feats = small_model.extract_features(dummy_batch)
        # ResNet-18 sort 512 features
        assert feats.shape[0] == 4
        assert feats.ndim == 2

    def test_no_normalize_output(self, dummy_batch: torch.Tensor) -> None:
        model = FoodEmbeddingModel(
            backbone_name="resnet18",
            pretrained=False,
            embedding_dim=32,
            normalize_output=False,
        )
        model.eval()
        out = model(dummy_batch)
        norms = out.norm(dim=1)
        # Sans normalisation, les normes ne devraient PAS toutes valoir 1
        assert not torch.allclose(norms, torch.ones(4), atol=1e-2)


# ---------------------------------------------------------------------------
# Gel / dégel
# ---------------------------------------------------------------------------

class TestFreeze:
    def test_freeze_backbone_no_grad(self) -> None:
        model = FoodEmbeddingModel(
            backbone_name="resnet18",
            pretrained=False,
            freeze_backbone=True,
        )
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_projection_head_trainable_when_frozen(self) -> None:
        model = FoodEmbeddingModel(
            backbone_name="resnet18",
            pretrained=False,
            freeze_backbone=True,
        )
        for param in model.projection.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self) -> None:
        model = FoodEmbeddingModel(
            backbone_name="resnet18",
            pretrained=False,
            freeze_backbone=True,
        )
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Sérialisation
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_and_load(
        self, small_model: FoodEmbeddingModel, tmp_path: Path, dummy_batch: torch.Tensor
    ) -> None:
        ckpt = tmp_path / "model.pt"
        small_model.save(ckpt)
        assert ckpt.exists()

        loaded = FoodEmbeddingModel.load(ckpt)
        loaded.eval()
        small_model.eval()

        with torch.no_grad():
            out_orig = small_model(dummy_batch)
            out_loaded = loaded(dummy_batch)

        assert torch.allclose(out_orig, out_loaded, atol=1e-6), \
            "Les sorties du modèle chargé diffèrent de l'original"

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            FoodEmbeddingModel.load("/nonexistent/path/model.pt")


# ---------------------------------------------------------------------------
# FoodImageDataset
# ---------------------------------------------------------------------------

class TestFoodImageDataset:
    def test_len(self, tmp_image_dir: Path) -> None:
        ds = FoodImageDataset(root_dir=tmp_image_dir)
        assert len(ds) == 6

    def test_item_keys(self, tmp_image_dir: Path) -> None:
        ds = FoodImageDataset(root_dir=tmp_image_dir)
        item = ds[0]
        assert "image" in item
        assert "path" in item

    def test_image_tensor_shape(self, tmp_image_dir: Path) -> None:
        ds = FoodImageDataset(root_dir=tmp_image_dir)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)

    def test_labels_detected(self, tmp_image_dir: Path) -> None:
        ds = FoodImageDataset(root_dir=tmp_image_dir)
        item = ds[0]
        assert "label" in item
        assert item["label"] in {0, 1}

    def test_classes_found(self, tmp_image_dir: Path) -> None:
        ds = FoodImageDataset(root_dir=tmp_image_dir)
        assert set(ds.classes) == {"pizza", "salade"}

    def test_missing_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            FoodImageDataset(root_dir="/nonexistent/path")


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_inference_transform_output_shape(self) -> None:
        img = Image.new("RGB", (300, 400))
        t = get_inference_transforms(image_size=224)
        tensor = t(img)
        assert tensor.shape == (3, 224, 224)

    def test_train_transform_output_shape(self) -> None:
        img = Image.new("RGB", (300, 400))
        t = get_train_transforms(image_size=224)
        tensor = t(img)
        assert tensor.shape == (3, 224, 224)

    def test_inference_is_deterministic(self) -> None:
        img = Image.new("RGB", (300, 400), color=(100, 150, 200))
        t = get_inference_transforms()
        t1 = t(img)
        t2 = t(img)
        assert torch.allclose(t1, t2)


# ---------------------------------------------------------------------------
# EmbeddingGenerator — sauvegarde
# ---------------------------------------------------------------------------

class TestEmbeddingGeneratorSave:
    @pytest.fixture()
    def dummy_data(self) -> dict:
        rng = np.random.default_rng(42)
        return {
            "embeddings": rng.random((10, 64), dtype=np.float32),
            "paths": np.array([f"img_{i}.jpg" for i in range(10)], dtype=object),
            "labels": np.arange(10, dtype=np.int32),
        }

    def test_save_hdf5(self, dummy_data: dict, tmp_path: Path) -> None:
        from src.process.embed_images import EmbeddingGenerator
        import h5py

        out = EmbeddingGenerator.save(dummy_data, tmp_path, fmt="hdf5")
        assert out.exists()
        with h5py.File(out, "r") as f:
            assert f["embeddings"].shape == (10, 64)
            assert len(f["paths"]) == 10

    def test_save_npy(self, dummy_data: dict, tmp_path: Path) -> None:
        from src.process.embed_images import EmbeddingGenerator

        out = EmbeddingGenerator.save(dummy_data, tmp_path, fmt="npy")
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape == (10, 64)

    def test_save_parquet(self, dummy_data: dict, tmp_path: Path) -> None:
        from src.process.embed_images import EmbeddingGenerator
        import pandas as pd

        out = EmbeddingGenerator.save(dummy_data, tmp_path, fmt="parquet")
        assert out.exists()
        df = pd.read_parquet(out)
        assert "e_0" in df.columns
        assert "path" in df.columns
        assert len(df) == 10

    def test_save_unknown_format_raises(self, dummy_data: dict, tmp_path: Path) -> None:
        from src.process.embed_images import EmbeddingGenerator

        with pytest.raises(ValueError, match="Format inconnu"):
            EmbeddingGenerator.save(dummy_data, tmp_path, fmt="csv")
