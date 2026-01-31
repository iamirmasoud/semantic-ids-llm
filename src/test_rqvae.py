#!/usr/bin/env python3
"""
Test script to run RQ-VAE training on synthetic/random embeddings.
Useful for verifying the training pipeline works without real data.

Usage:
    uv run -m src.test_rqvae
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from src.logger import setup_logger
from src.train_rqvae import RQVAE, RQVAEConfig, EmbeddingDataset, train_rqvae

logger = setup_logger("test-rqvae", log_to_file=False)


@dataclass
class TestConfig:
    """Configuration for test run."""

    n_items: int = 1000  # Number of synthetic items
    embedding_dim: int = 1024  # Embedding dimension
    n_clusters: int = 10  # Number of clusters to simulate (for more realistic data)
    noise_scale: float = 0.1  # Noise scale for cluster perturbation
    seed: int = 42  # Random seed for reproducibility


def generate_synthetic_embeddings(config: TestConfig) -> tuple[list[str], np.ndarray]:
    """
    Generate synthetic embeddings with cluster structure.
    This simulates real item embeddings better than pure random noise.

    Returns:
        Tuple of (item_ids, embeddings)
    """
    np.random.seed(config.seed)

    # Generate cluster centers (normalized)
    cluster_centers = np.random.randn(config.n_clusters, config.embedding_dim)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    # Assign items to clusters
    cluster_assignments = np.random.randint(0, config.n_clusters, config.n_items)

    # Generate embeddings as perturbed cluster centers
    embeddings = []
    for i in range(config.n_items):
        center = cluster_centers[cluster_assignments[i]]
        noise = np.random.randn(config.embedding_dim) * config.noise_scale
        embedding = center + noise
        # L2 normalize (like real embeddings)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Generate fake ASINs
    item_ids = [f"TEST_{i:06d}" for i in range(config.n_items)]

    logger.info(f"Generated {config.n_items} synthetic embeddings with {config.n_clusters} clusters")
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Embedding norms (should be ~1.0): mean={np.linalg.norm(embeddings, axis=1).mean():.4f}")

    return item_ids, embeddings


def create_test_parquet(item_ids: list[str], embeddings: np.ndarray, output_path: Path) -> None:
    """Create a parquet file with the expected schema."""
    df = pl.DataFrame({
        "parent_asin": item_ids,
        "embedding": embeddings.tolist(),
    })

    df.write_parquet(output_path)
    logger.info(f"Saved test data to {output_path}")
    logger.info(f"Schema: {df.schema}")


def run_test_training():
    """Run a quick RQ-VAE training on synthetic data."""
    # Create synthetic data
    test_config = TestConfig(
        n_items=1000,  # Small dataset for quick test
        embedding_dim=1024,
        n_clusters=10,
        noise_scale=0.1,
    )

    item_ids, embeddings = generate_synthetic_embeddings(test_config)

    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save test parquet
        test_parquet_path = tmpdir / "test_embeddings.parquet"
        create_test_parquet(item_ids, embeddings, test_parquet_path)

        # Configure RQ-VAE for quick test
        rqvae_config = RQVAEConfig(
            # Data settings
            embeddings_path=test_parquet_path,
            checkpoint_dir=tmpdir / "checkpoints",

            # Model parameters (smaller for testing)
            item_embedding_dim=1024,
            encoder_hidden_dims=[256, 128],  # Smaller encoder
            codebook_embedding_dim=32,
            codebook_quantization_levels=3,
            codebook_size=64,  # Smaller codebooks for test
            commitment_weight=0.25,
            use_rotation_trick=True,

            # Training parameters (quick test)
            batch_size=256,
            gradient_accumulation_steps=1,
            num_epochs=50,  # Few epochs for test
            scheduler_type="cosine_with_warmup",
            warmup_start_lr=1e-8,
            warmup_steps=10,
            max_lr=3e-4,
            min_lr=1e-6,
            use_gradient_clipping=True,
            gradient_clip_norm=1.0,
            use_kmeans_init=True,
            reset_unused_codes=True,
            steps_per_codebook_reset=5,
            val_split=0.1,

            # Logging
            steps_per_train_log=5,
            steps_per_val_log=20,
        )

        rqvae_config.log_config()

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

        # Load dataset
        dataset = EmbeddingDataset(str(test_parquet_path))

        # Split into train/val
        val_size = int(len(dataset) * rqvae_config.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=rqvae_config.batch_size,
            shuffle=True,
            num_workers=0,  # No multiprocessing for test
            drop_last=False,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=rqvae_config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Create model
        model = RQVAE(rqvae_config)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Initialize wandb in offline mode for testing
        import wandb
        wandb.init(mode="offline", project="rqvae-test", name="test-run")

        # Train!
        logger.info("Starting training...")
        train_rqvae(
            model=model,
            data_loader=train_loader,
            val_loader=val_loader,
            config=rqvae_config,
            device=device,
        )

        # Test encoding to semantic IDs
        logger.info("\n=== Testing Semantic ID Generation ===")
        model.eval()
        with torch.no_grad():
            # Get a sample batch
            sample_batch = embeddings[:10]
            sample_tensor = torch.tensor(sample_batch, dtype=torch.float32).to(device)

            # Encode to semantic IDs
            semantic_ids = model.encode_to_semantic_ids(sample_tensor)
            logger.info(f"Semantic IDs shape: {semantic_ids.shape}")
            logger.info(f"Sample semantic IDs:\n{semantic_ids}")

            # Test reconstruction
            reconstructed = model.decode_from_semantic_ids(semantic_ids)
            logger.info(f"Reconstructed shape: {reconstructed.shape}")

            # Calculate reconstruction error
            recon_error = F.mse_loss(reconstructed.cpu(), sample_tensor.cpu())
            logger.info(f"Reconstruction MSE: {recon_error.item():.4f}")

            # Check unique IDs proportion
            unique_prop = model.calculate_unique_ids_proportion(semantic_ids)
            logger.info(f"Unique IDs proportion: {unique_prop:.1%}")

        logger.info("\n=== Test Complete! ===")
        wandb.finish()


if __name__ == "__main__":
    run_test_training()
