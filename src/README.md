# Source Code Overview

## tokenize_items.py

Pre-tokenizes product text data to separate CPU-bound tokenization from GPU inference.

**Input:** 
- File: `data/output/Video_Games_items_updated.parquet`
- Columns used: `item_context`

**Process:**
1. Wraps each product description with an instruction prefix for embedding task guidance
2. Tokenizes text using Qwen3-Embedding tokenizer with padding and truncation
3. Batches the tokenization for efficient processing

**Output:** 
- File: `data/output/Video_Games_tokenized.npz`
- Contents: `input_ids`, `attention_mask`, `max_length`, `n_items`

---

## embed_items.py

Generates dense vector embeddings for products using the pre-tokenized data.

**Input:** 
- File: `data/output/Video_Games_tokenized.npz` (from tokenize_items.py)
- File: `data/output/Video_Games_items_updated.parquet`
- Columns used: `item_context` (for verification/logging)

**Process:**
1. Loads the Qwen3-Embedding model onto GPU/MPS/CPU
2. Uses a DataLoader with pinned memory for efficient GPU transfers
3. Performs last-token pooling to extract embeddings from model hidden states
4. Truncates embeddings to target dimension (1024)
5. L2-normalizes all embeddings for consistent similarity comparisons

**Output:** 
- File: `data/output/Video_Games_items_with_embeddings.parquet`
- Columns added: `embedding` (List[Float32], dim=1024)

---

## train_rqvae.py

Trains a Residual Quantized VAE (RQ-VAE) to encode item embeddings into discrete hierarchical semantic IDs.

**Input:**
- File: `data/output/Video_Games_items_with_embeddings.parquet` (from embed_items.py)
- Columns used: `embedding` (1024-dim vectors), `parent_asin`

**Model Architecture:**
- **Encoder:** MLP that projects 1024-dim embeddings → hidden layers [512, 256, 128] → 32-dim latent space
- **Decoder:** Mirror of encoder, reconstructs original embedding from quantized latent
- **Residual Quantization:** Multiple levels (default 3) of vector quantization applied sequentially to residuals
- **Codebook:** Each level has 256 codes of 32 dimensions

**Key Components:**
1. `RQVAEConfig`: Dataclass with all hyperparameters (model dims, training params, logging intervals)
2. `EmbeddingDataset`: Loads embeddings from parquet into PyTorch tensors
3. `VectorQuantizer`: Standard VQ with learnable codebooks and commitment loss
4. `EMAVectorQuantizer`: Alternative VQ using exponential moving average updates (optional)
5. `RQVAE`: Main model combining encoder, residual quantization layers, and decoder

**Training Features:**
- **K-means initialization:** Codebooks initialized via k-means clustering on encoded data
- **Rotation trick:** Improved gradient flow through quantization (from arXiv:2410.06424)
- **Codebook reset:** Unused codes periodically re-initialized from batch samples
- **Gradient accumulation:** Support for effective batch sizes larger than memory allows
- **LR scheduling:** Cosine annealing with optional warmup phase
- **Checkpointing:** Best model saved based on validation loss

**Loss Function:**
- `total_loss = reconstruction_loss + vq_loss`
- `vq_loss = codebook_loss + β × commitment_loss` (summed across all quantization levels)

**Metrics Tracked:**
- Codebook usage rate per level (proportion of codes being used)
- Unique IDs proportion (how many items get distinct semantic IDs)
- Average residual norm (quantization quality indicator)
- Gradient norms (before/after clipping)

**Output:**
- Checkpoints: `checkpoints/rqvae/checkpoint_step_N.pth`, `best_model.pth`, `final_model.pth`
- W&B artifacts: Best model uploaded with metadata (val_loss, codebook usage, etc.)

**Default Configuration:**
```python
codebook_quantization_levels = 3   # Hierarchical depth
codebook_size = 256                # Codes per level
codebook_embedding_dim = 32        # Latent dimension
batch_size = 32768                 # Large batch for codebook stability
num_epochs = 20000                 # Training iterations
max_lr = 3e-4                      # Peak learning rate
```

---

## Execution Order

```
tokenize_items.py → embed_items.py → train_rqvae.py
```

1. **tokenize_items.py** (CPU-intensive): Pre-tokenize product text
2. **embed_items.py** (GPU-intensive): Generate dense embeddings
3. **train_rqvae.py** (GPU-intensive): Learn hierarchical semantic IDs from embeddings

This separation allows for better resource utilization and easier debugging.
