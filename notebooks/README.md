# Notebooks

This directory contains Jupyter notebooks for the Semantic ID generation pipeline. The notebooks are designed to be run sequentially, with each notebook building on the outputs of the previous ones.

## Pipeline Overview

```
01-prep-items-and-sequences.ipynb
         │
         ▼
┌────────┴────────┐
│                 │
▼                 ▼
02-clean-descriptions.ipynb    03-clean-titles.ipynb
│                 │
└────────┬────────┘
         │
         ▼
04-augment-metadata.ipynb
         │
         ▼
05-update-items-and-sequences.ipynb
         │
         ▼
06-get-semantic-ids-per-asin.ipynb
         │
         ▼
07-get-semantic-ids-to-asin-sequences.ipynb
         │
         ▼
08-prep-finetuning-data.ipynb
         │
         ▼
┌────────┴────────┐
│                 │
▼                 ▼
09-evaluate-sasrec-baseline.ipynb    10-evaluate-sasrec-semantic.ipynb
```

---

## 01 - Data Preparation for Semantic ID Generation

**Purpose:** Download and prepare the Amazon dataset for semantic ID generation.

### Configuration
- `CATEGORY = "Video_Games"` (or `"Baby_Products"`)
- `MIN_SEQUENCE_LENGTH = 3`
- `MAX_SEQUENCE_LENGTH = 100`

### Steps

#### 1. Download and Load Data
- Download 3 data files from Amazon 2023 dataset (McAuley Lab):
  - Item metadata: `meta_{CATEGORY}.jsonl.gz`
  - Review data: `review_{CATEGORY}.jsonl.gz`
  - Sequences data: `{CATEGORY}.train.csv.gz`
- Unzip and save to local `data/` directory

#### 2. Prepare Item Metadata
- Load item metadata from JSONL using Polars
- Process list columns into text fields:
  - `description_text`: Join description list with spaces
  - `features_text`: Join features list with spaces
  - `categories_text`: Join categories with " > " separator
- **Filter items** requiring:
  - Title length > 20 characters
  - Description length > 100 characters
- Fill null values for all metadata fields
- Create `item_context` - a concatenated string combining:
  ```
  Product: {title}
  Description: {description_text}
  Features: {features_text}
  Category: {main_category}, Category tree: {categories_text}
  Store: {store}
  Average rating: {average_rating}, Rating count: {rating_number}
  Price: {price}
  ```

#### 3. Load User Sequences
- Load CSV with columns: `user_id`, `parent_asin`, `rating`, `timestamp`, `history`
- **Deduplicate by user_id** - keep row with longest history per user (reduces ~736K rows to ~91K)
- Create `sequence` column by appending target item (`parent_asin`) to history list

#### 4. Filter Items Without Metadata
- Remove items from sequences that don't exist in valid items set
- Filter out sequences shorter than `MIN_SEQUENCE_LENGTH`
- Typical reduction: ~14% of rows removed

#### 5. Truncate Long Sequences
- Truncate sequences longer than `MAX_SEQUENCE_LENGTH` (keep last N items)
- Calculate sequence length statistics (mean ~6.5, median ~5.0)

#### 6. Save Processed Data
- `{CATEGORY}_sequences.parquet`: user_id, sequence (list), sequence_length
- `{CATEGORY}_items.parquet`: Full item metadata with item_context

---

## 02 - Clean Item Descriptions and Features

**Purpose:** Use Gemini API to clean and improve product descriptions.

### Configuration
- `MODEL = "gemini-2.5-flash-lite"`
- `MAX_RETRIES = 5`
- `RETRY_DELAY = 1.0` seconds
- `RATE_LIMIT_DELAY = 0.2` seconds between API calls
- `MAX_THREADS = 10`
- `MAX_DESCRIPTION_LENGTH = 2000` (truncate before API call)
- `MIN_DESCRIPTION_LENGTH = 50`

### Steps

#### 1. Load Data and Initial Analysis
- Load item data from `{CATEGORY}_items.parquet`
- Analyze text quality issues:
  - Length statistics (min, max, mean, median)
  - Count of empty/truncated descriptions
  - HTML entities count (`&amp;`, `&lt;`, `&gt;`, `&nbsp;`)
  - Excessive whitespace patterns
  - Features without punctuation
- Visualize text length distributions with histograms

#### 2. Initialize Gemini API Client
- Create Gemini client using Google GenAI SDK
- Test API connection with simple prompt

#### 3. Define Cleaning Functions
- Detailed prompt instructs Gemini to:
  1. Fix grammar, spelling, and punctuation errors
  2. Remove HTML entities (like `&amp;`, `&lt;`, `&gt;`)
  3. Fix truncated sentences (ending with `...`)
  4. Remove excessive whitespace and formatting artifacts
  5. Keep all important product information
  6. Maintain a professional, informative tone
  7. Complete cut-off descriptions naturally
  8. Convert bullet-point lists into coherent, readable text
  9. Group related features together logically
  10. Remove redundant information
  - **Keep cleaned description under 200 words**
- Response extracted from `<clean_description>` tags

#### 4. Parallel Processing with Checkpointing
- Setup checkpoint file: `{CATEGORY}_descriptions_clean.csv`
- Load already-processed ASINs to avoid reprocessing
- Filter to only unprocessed items
- Use `ThreadPoolExecutor` with thread-safe CSV writing (`csv_lock`)
- Process items in parallel:
  - Truncate description if > `MAX_DESCRIPTION_LENGTH`
  - Call `clean_description()` function
  - Write immediately to checkpoint file after each item
  - Rate limiting between API calls
- Track success/failure counts with progress bar (tqdm)
- Return "NA" after all retry attempts fail

#### 5. Verify and Load Results
- Load cleaned descriptions from checkpoint CSV
- Display sample of cleaned data for verification

---

## 03 - Clean Product Titles

**Purpose:** Use Gemini API to clean and standardize product titles.

### Configuration
- `MODEL = "gemini-2.5-flash-lite"`
- `BATCH_SIZE = 10`
- `MAX_RETRIES = 5`
- `RETRY_DELAY = 1.0` seconds
- `RATE_LIMIT_DELAY = 0.1` seconds (faster for shorter text)
- `MAX_THREADS = 15` (more threads for shorter text)
- `MAX_TITLE_LENGTH = 200`
- `MIN_TITLE_LENGTH = 10`

### Steps

#### 1. Load Data and Analyze Titles
- Load item data from parquet
- Analyze title quality issues:
  - Length statistics (min ~21, max ~620, mean ~76)
  - All caps titles count
  - Excessive punctuation (`!!`, `??`, `...`)
  - Promotional text (`NEW!`, `SALE`, `LIMITED`, `EXCLUSIVE`)
  - Special characters (`★`, `☆`, `♥`, `©`, `®`, `™`)
  - Parentheses overuse (≥3 parentheses)
  - Platform keywords presence
- Visualize title length distribution

#### 2. Initialize Gemini API Client
- Connect and test API

#### 3. Define Title Cleaning Functions
- Prompt instructs Gemini to:
  1. Remove promotional text (NEW!, SALE!, LIMITED!, etc.)
  2. Fix capitalization - use proper title case
  3. Remove excessive punctuation and special characters
  4. Remove duplicate or redundant information
  5. Keep important details: brand names, model numbers, key specifications
  6. Remove seller/store information
  7. Fix spelling and grammar errors
  8. Standardize abbreviations (e.g., "Ed." → "Edition", "w/" → "with")
  9. Remove platform redundancy if already clear from context
  10. **Keep title under 200 characters**
- Response extracted from `<clean_title>` tags

#### 4. Parallel Title Processing with Checkpointing
- Checkpoint file: `{CATEGORY}_titles_clean.csv`
- Same parallel processing pattern as descriptions
- Saves original title if processing fails

#### 5. Verify and Analyze Cleaned Titles
- Load and display cleaned titles
- Check for remaining XML tags

---

## 04 - Extract Product Metadata

**Purpose:** Extract structured product metadata using Gemini API.

### Configuration
- `MODEL = "gemini-2.5-flash-lite"`
- `MAX_THREADS = 15`
- `MAX_RETRIES = 5`

### Steps

#### 1. Load and Merge Data
- Load original item metadata
- Load cleaned descriptions (filter out "NA")
- Load cleaned titles (filter out "NA")
- Inner join all three datasets
- Clean up remaining XML tags with regex

#### 2. Create Item Context
- Rename columns (original → cleaned versions)
- Build comprehensive context string:
  ```
  Title: {clean_title}
  Description: {clean_description}
  Category: {categories_text}
  Features: {features_text}
  ```

#### 3. Define Metadata Extraction Functions
- JSON extraction prompt for fields:
  - `product_type`: Game | Hardware | Accessory | DLC | Bundle | other
  - `platform`: Array of platforms (PS5, PS4, Xbox Series X, Xbox One, Nintendo Switch, PC, Steam Deck, etc.)
  - `genre`: Array of genres with subgenres:
    - Main: Action, RPG, Strategy, Sports, Racing, Puzzle, Adventure
    - Subgenres: Roguelike, Soulslike, Metroidvania, Battle Royale, MOBA, Auto Battler, Gacha, Idle, Tower Defense
    - Thematic: Open World, Sandbox, Survival, Horror
  - `hardware_type`: Console | Controller | Headset | Cable | Storage | Skin | Stand | Cooling | Capture Card
  - `brand`: Primary manufacturer or brand name
  - `multiplayer`: Single-player | Local Co-op | Online Multiplayer | MMO | MMORPG | PvP | PvE | Couch Co-op
  - `franchise`: Game series or franchise name
- Parse JSON response, clean markdown code blocks

#### 4. Parallel Metadata Extraction with Checkpointing
- Checkpoint file: `{CATEGORY}_metadata.jsonl` (JSONL format for structured data)
- Thread-safe file writing with lock
- Write each result as JSON line immediately
- Track extraction failures separately

#### 5. Analyze Metadata Coverage
- Field coverage statistics (e.g., genre: 95%, platform: 90%)
- Product type distribution (Game, Accessory, Hardware, etc.)
- Platform distribution (flattened from arrays)
- Genre distribution (top 20)
- Multiplayer mode distribution

#### 6. Save as DataFrame
- Normalize records (ensure consistent list types)
- Convert to Polars DataFrame
- Save as `{CATEGORY}_metadata_extracted.parquet`

---

## 05 - Update Items and Sequences

**Purpose:** Merge all cleaned and augmented data into final datasets.

### Steps

#### 1. Load and Merge All Data
- Original item metadata (`{CATEGORY}_items.parquet`)
- Augmented metadata (`{CATEGORY}_metadata_extracted.parquet`)
- Cleaned descriptions (`{CATEGORY}_descriptions_clean.csv`)
- Cleaned titles (`{CATEGORY}_titles_clean.csv`)
- Inner join on `parent_asin`

#### 2. Clean Remaining XML Tags
- Remove all `<tag>` patterns with regex
- Collapse multiple spaces
- Strip whitespace

#### 3. Description Length Analysis
- Calculate character lengths for original vs. cleaned
- Statistics: mean, median, std deviation
- Reduction metrics (absolute and percentage)
- Total character reduction across dataset (~44% reduction typical)

#### 4. Update Item Context
- Rebuild `item_context` with cleaned title and description:
  ```
  Title: {clean_title}
  Description: {clean_description}
  Category: {categories_text}
  Features: {features_text}
  ```

#### 5. Update Sequences
- Load original sequences
- Filter to keep only items with valid metadata
- Re-filter sequences by minimum length

#### 6. Save Updated Data
- Updated items parquet with all cleaned/augmented fields
- Updated sequences parquet with filtered sequences

---

## 06 - Get Semantic IDs Per ASIN

**Purpose:** Generate semantic IDs for each product using trained RQ-VAE model.

### Configuration
- `artifact_path`: W&B artifact path for RQ-VAE model (e.g., `eugeneyan/rqvae/rqvae-best-{CATEGORY}:v1882`)
- RQ-VAE Config:
  - `item_embedding_dim = 1024`
  - `encoder_hidden_dims = [512, 256, 128]`
  - `codebook_embedding_dim = 32`
  - `codebook_quantization_levels = 3`
  - `codebook_size = 256`

### Steps

#### 1. Download Model from W&B
- Initialize wandb run
- Download artifact to `models/` directory
- Rename to versioned filename

#### 2. Load Model
- Load RQ-VAE config from checkpoint
- Initialize RQVAE model
- Load state dict weights
- Set to eval mode

#### 3. Load Item Embeddings
- Load pre-computed embeddings from `{CATEGORY}_items_with_embeddings.parquet`
- Embeddings are 1024-dimensional vectors from text encoder

#### 4. Generate Semantic IDs
- Pass embeddings through RQ-VAE encoder
- Quantize through multiple codebook levels
- Get indices from each quantization level
- Format as hierarchical ID string:
  ```
  <|sid_start|><|sid_{level0}|><|sid_{level1+256}|><|sid_{level2+512}|><|sid_{level3+768}|><|sid_end|>
  ```

#### 5. Analyze Distribution
- Codebook utilization per level
- ID uniqueness statistics
- Collision analysis

#### 6. Save Semantic IDs
- Output: `{CATEGORY}_semantic_ids.parquet`
- Columns: `parent_asin`, `semantic_id`, individual level indices

---

## 07 - Map ASINs and Sequences to Semantic IDs

**Purpose:** Convert user sequences from ASINs to semantic IDs.

### Configuration
- `RQVAE_VERSION`: Version of RQ-VAE model to use

### Steps

#### 1. Download Model from W&B
- Fetch same RQ-VAE model used in notebook 06

#### 2. Load Semantic ID Mappings
- Load `{CATEGORY}_semantic_ids.parquet`
- Create ASIN → semantic_id lookup dictionary

#### 3. Load User Sequences
- Load `{CATEGORY}_sequences.parquet`
- Each row has user_id and list of ASINs

#### 4. Map Sequences to Semantic IDs
- For each sequence, convert ASINs to semantic IDs
- Handle missing mappings (items filtered out earlier)
- Create parallel columns: `sequence` (ASINs) and `semantic_sequence` (semantic IDs)

#### 5. Create Train/Val Split
- Split sequences for model training (e.g., 95/5)
- Ensure no user overlap between splits

#### 6. Verify Mapping
- Check coverage percentage
- Identify any unmapped items

#### 7. Save Mapped Sequences
- `{CATEGORY}_sequences_with_semantic_ids.parquet` (full)
- `{CATEGORY}_sequences_with_semantic_ids_train.parquet`
- `{CATEGORY}_sequences_with_semantic_ids_val.parquet`

---

## 08 - Prepare Pre-training Data for Semantic LLM

**Purpose:** Generate exhaustive training data for teaching LLMs to work with semantic IDs.

### Configuration
```python
SYSTEM_PROMPT = """
You are a helpful AI assistant that understands and works with semantic IDs for product recommendations. 

Semantic IDs are hierarchical identifiers in the format <|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|> that encode product relationships and categories.
"""
```

### Steps

#### 1. Load Data
- Items with metadata (`{CATEGORY}_items.parquet`)
- Semantic ID mappings (`{CATEGORY}_semantic_ids.parquet`)
- User sequences (`{CATEGORY}_sequences_with_semantic_ids.parquet`)
- Join items with semantic IDs

#### 2. Generate Type A Data (SemanticID → Text)
For every item, generate multiple variations:
- **ID to title**: `"Product {semantic_id} has title:"` → `{title}`
- **ID to description**: `"Item {semantic_id} is described as:"` → `{description}`
- **ID to category**: `"Product {semantic_id} belongs to category:"` → `{category}`
- **ID to features**: `"The features of {semantic_id} are:"` → `{features}`

#### 3. Generate Type B Data (Text → SemanticID)
- **Title to ID**: `"What is the semantic ID for '{title}'?"` → `{semantic_id}`
- **Description to ID**: `"Find the product ID for: {description}"` → `{semantic_id}`

#### 4. Generate Type C Data (Sequence Prediction)
- Given user history, predict next item
- Multiple sequence lengths (3, 5, 10 items)
- Various prompt formats:
  - `"User purchased {seq}. Recommend next:"` → `{next_item}`
  - `"Based on history {seq}, predict:"` → `{next_item}`

#### 5. Generate Type D Data (Similarity/Relationship)
- Items sharing codebook levels (similar semantic IDs)
- Category relationships
- Franchise groupings

#### 6. Format for Training
- Create instruction-following format
- Add system prompts
- Balance across data types

#### 7. Save Training Data
- Export to `semantic_llm_training/` directory
- Multiple JSONL files by data type
- Combined training file

---

## 09 - Evaluate SASRec Baseline

**Purpose:** Evaluate the baseline SASRec model using traditional item IDs.

### Configuration
```python
@dataclass
class EvalConfig:
    dataset: str = "Video_Games"
    sequences_path: Path = "data/output/Video_Games_sequences_with_semantic_ids_val.parquet"
    artifact_path: str = "eugeneyan/sasrec-experiments/sasrec-best-Video_Games:v85"
    val_fraction: float = 1.0
    batch_size: int = 256
    num_negative_samples: int = 1000
    max_seq_length: int = 50
    seed: int = 42
```

### Steps

#### 1. Download Model from W&B
- Fetch trained SASRec checkpoint
- Load model config and weights

#### 2. Load Validation Data
- Load held-out user sequences
- Create item vocabulary mapping

#### 3. Initialize Model
- Load SASRec architecture
- Set to eval mode
- Move to device (GPU/MPS)

#### 4. Evaluation Loop
For each validation sequence:
- Extract history (all items except last)
- Target = last item in sequence
- Generate embeddings for history
- Score target against negative samples
- Compute ranking metrics

#### 5. Compute Metrics
- **Hit Rate @ K**: Proportion where target in top K
- **NDCG @ K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- K values: 1, 5, 10, 20

#### 6. Analysis
- Breakdown by sequence length
- Performance by item popularity
- Error analysis

---

## 10 - Evaluate Semantic SASRec Model

**Purpose:** Evaluate the semantic ID-based SASRec model and compare with baseline.

### Configuration
```python
@dataclass
class EvalConfig:
    dataset: str = "Video_Games"
    sequences_path: Path = "data/output/Video_Games_sequences_with_semantic_ids_val.parquet"
    artifact_path: str = "eugeneyan/sasrec-experiments/semantic-sasrec-final-Video_Games:v0"
    num_levels: int = 4  # Hierarchical levels in semantic ID
    codebook_size: int = 256  # Size per level
    val_fraction: float = 1.0
    batch_size: int = 256
    num_negative_samples: int = 1000
    max_seq_length: int = 50
```

### Steps

#### 1. Download Model from W&B
- Fetch trained SemanticSASRec checkpoint

#### 2. Load Validation Data
- Load sequences with semantic IDs
- Use `encode_semantic_id()` function for tokenization

#### 3. Initialize Model
- Load SemanticSASRec architecture (handles hierarchical IDs)
- Set to eval mode

#### 4. Evaluation Loop
For each validation sequence:
- Encode semantic ID sequence through levels
- Predict next semantic ID
- Score against negative semantic IDs
- Compute ranking metrics

#### 5. Compute Metrics
- Same metrics as baseline: Hit Rate, NDCG, MRR at K=1,5,10,20

#### 6. Compare with Baseline
- Side-by-side metric comparison
- Analyze where semantic IDs help/hurt
- Cold-start item performance
- Generalization to unseen items

---

## Output Files Summary

| Notebook | Key Outputs |
|----------|-------------|
| 01 | `{CATEGORY}_items.parquet` (66K items), `{CATEGORY}_sequences.parquet` (78K sequences) |
| 02 | `{CATEGORY}_descriptions_clean.csv` (~44% character reduction) |
| 03 | `{CATEGORY}_titles_clean.csv` (standardized titles) |
| 04 | `{CATEGORY}_metadata.jsonl`, `{CATEGORY}_metadata_extracted.parquet` |
| 05 | Updated items and sequences with all cleaning/augmentation merged |
| 06 | `{CATEGORY}_semantic_ids.parquet` (ASIN → semantic ID mapping) |
| 07 | `{CATEGORY}_sequences_with_semantic_ids.parquet` (train/val splits) |
| 08 | Training JSONL files in `semantic_llm_training/` (~318K Type A samples, etc.) |
| 09-10 | Evaluation metrics (Hit@K, NDCG@K, MRR) |

---

## Requirements

- **API Keys**: 
  - Gemini API key (set in `.env` as `GEMINI_API_KEY`)
- **W&B**: 
  - Weights & Biases account for model artifact management
  - Login via `wandb login`
- **Compute**: 
  - GPU/MPS recommended for embedding generation and model evaluation
  - CPU sufficient for data processing notebooks (01-05, 07-08)
- **Dependencies**: 
  - See `pyproject.toml` for full list
  - Key packages: `polars`, `torch`, `transformers`, `google-genai`, `wandb`, `tqdm`
