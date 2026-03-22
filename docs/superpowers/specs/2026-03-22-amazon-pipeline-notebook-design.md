# Amazon End-to-End Pipeline Notebook — Design Spec

## Overview

A new Jupyter notebook (`Amazon_End_to_End_Pipeline.ipynb`) that demonstrates the full graph ML pipeline — from raw data to trained GNN — using a real Amazon co-purchasing dataset. The notebook proves the repo's central thesis: no single resource teaches how to go from a graph database to a GNN. This notebook does it end to end, nothing skipped.

**Task:** Link prediction — predict which Amazon products will be co-purchased.

**Dataset:** Amazon product co-purchasing network from SNAP (Stanford Network Analysis Platform). 334,863 product nodes, 925,872 co-purchase edges, 75,149 ground-truth product communities.

**Scope:** ~25–30 cells. Simple code, no theory rehash. Points to the main guide for GNN theory.

## Data Files

| File | Contents |
|------|----------|
| `data/com-Amazon.mtx` | Co-purchasing graph (334,863 x 334,863, 925,872 edges) |
| `data/com-Amazon_Communities_all.mtx` | All ground-truth community assignments (75,149 communities) |
| `data/com-Amazon_Communities_top5000.mtx` | Top 5,000 communities by quality |
| `data/com-Amazon_nodeid.mtx` | Original Amazon product ID mapping |

All files are Matrix Market (.mtx) format, loaded with `scipy.io.mmread`.

## Notebook Sections

### Section 1: Introduction & Data Exploration (5–6 cells)

**Purpose:** Orient the reader on what the dataset is and what the notebook will do.

**Markdown content:**
- Notebook purpose: full pipeline with a real dataset (raw data → Neo4j → GNN → predictions)
- Dataset description: each node is an Amazon product, each edge means "customers who bought X also bought Y", communities represent product categories
- Source: SNAP Stanford

**Code cells:**
- Install & imports cell: all imports at the top (`scipy`, `networkx`, `neo4j`, `torch`, `torch_geometric`, `matplotlib`, `numpy`, `pandas`, `sklearn`). Set random seeds (`np.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)`) for reproducibility.
- Load `com-Amazon.mtx` via `scipy.io.mmread` → sparse matrix → NetworkX graph
- Load `com-Amazon_nodeid.mtx` to show original Amazon product ID mapping. Note in markdown: these are opaque numeric IDs (not human-readable product names) — the purpose is to show that a mapping exists back to the original Amazon catalog.
- Basic stats: node count, edge count, density, connected components
- Degree distribution: histogram + summary stats (min, max, mean, median degree)
- Visualize a small neighborhood: pick one node, plot its 2-hop subgraph with NetworkX spring layout

### Section 2: Subsampling (2–3 cells)

**Purpose:** Extract a laptop-friendly subset while preserving graph structure.

**Method:** BFS from a fixed seed node (deterministic, reproducible), expand until ~10,000 nodes are reached, take the induced subgraph. This preserves local community structure (unlike random node sampling, which fragments the graph).

**Code cells:**
- Subsample function using BFS from a fixed seed node → ~10K node connected subgraph
- Build a mapping from original node IDs to new 0..N-1 indices (required for PyG)
- Filter the community matrix (`Communities_all.mtx`) to only the sampled nodes, reindex rows to match the new node indices. This must happen here, before features are built in Section 4.
- Print comparison: subsampled vs original (nodes, edges, density)

### Section 3: Neo4j Ingestion & Querying (5–6 cells)

**Purpose:** Load the subgraph into Neo4j and explore it as a graph database. Self-contained — does not assume the main guide was completed.

**Markdown content:**
- Brief Neo4j setup instructions (download Neo4j Desktop, create a local DB, start it)
- Note: if you already set up Neo4j in the main guide, you're good to go. If you skipped it, that's completely fine — we walk through everything here.

**Code cells:**
- Install `neo4j` Python driver
- Connect to Neo4j, verify connection
- Clear existing data, batch-insert using `UNWIND $batch` Cypher pattern (not per-node inserts — that would be too slow for 10K+ nodes). Insert nodes as `(:Product {product_id, node_idx})` and edges as `[:CO_PURCHASED]`
- Print confirmation (nodes/edges inserted)
- Cypher exploration queries:
  - Count nodes and edges (verify ingestion)
  - Find the most connected product (highest degree)
  - Show a product's co-purchase neighborhood ("what else do buyers of product X buy?")

### Section 4: Neo4j → PyG Conversion (3–4 cells)

**Purpose:** The critical middle step — pull data from Neo4j and convert it into the numerical format a GNN expects.

**Markdown content:**
- Frame this as the step most tutorials skip
- Explain that since the dataset has no product attributes, we engineer features from graph structure itself — realistic for many real pipelines

**Code cells:**
- Query Neo4j via Cypher to pull all nodes and edges
- Engineer node features from graph structure (since the dataset has no product attributes):
  - **Degree** (normalized)
  - **Clustering coefficient**
  - **Number of communities** the node belongs to (scalar, from `Communities_all.mtx`)
  - **Community indicator vector**: keep only communities that have 2+ members in the subsample, drop the rest. This yields a compact binary vector (~20–50 dims depending on subsample) rather than a sparse 75K-dim vector. Explain in markdown why this filtering matters.
- Construct PyG `Data` object: `x` (feature matrix), `edge_index` (COO format)
- Print the PyG Data object to show final shape

### Section 5: Train/Test Split for Link Prediction (2–3 cells)

**Purpose:** Set up the link prediction task.

**Markdown content:**
- Explain the setup: hide some real edges (positive test), generate fake edges (negative samples), train the GNN to tell them apart

**Code cells:**
- Use PyG's `RandomLinkSplit` with 80/10/10 train/val/test split
- Print split stats: edges per set, negative samples
- Sanity check: confirm no test edges leak into training graph

### Section 6: GNN Model & Training (4–5 cells)

**Purpose:** Train a GCN for link prediction.

**Markdown content:**
- Brief recap of what the model does (no theory rehash, point to main guide Section 7)

**Code cells:**
- Define `LinkPredGCN`: 2-layer GCN encoder + dot-product decoder (~20 lines)
- Training loop: Adam optimizer, BCE loss, AUC on validation set every 10 epochs, ~200 epochs
- Training curve plot: loss and AUC over epochs (two-panel matplotlib)
- Print final test AUC

**Model architecture:**
- Input → GCNConv(in_features, 64) → ReLU → GCNConv(64, 32) → embeddings
- Decoder: dot product of source/destination embeddings
- Loss: binary cross-entropy
- Optimizer: Adam (lr=0.01)

### Section 7: Results & Visualization (3–4 cells)

**Purpose:** Show what the model learned and provide a visual payoff.

**Code cells:**
- t-SNE visualization of learned embeddings, colored by dominant community (the community with the most members in the node's membership set; nodes with no community assignment get a neutral "uncategorized" color)
- Top 10 predicted new co-purchase links (highest-scoring non-edges), displayed as a table showing both internal node indices and original Amazon product IDs (from the nodeid file)
- Closing markdown: one-paragraph recap of the full pipeline

## Style & Conventions

- Match the Knowledge_Graph notebook's style: PyTorch Geometric, pedagogical markdown before code
- Decorative code headers (`# ====== Step Name ======`)
- Print statements showing intermediate results
- Docstrings on model class
- No theory rehash — reference the main guide
- Hyperparameters stated explicitly in code

## Dependencies

```
scipy (mmread)
networkx
neo4j
torch
torch_geometric
matplotlib
numpy
pandas
scikit-learn (t-SNE)
```

## Position in Repository

| Notebook | Purpose |
|----------|---------|
| `Knowledge_Graph_GNN_Neo4j_Guide.ipynb` | Theory + logistics example |
| `GNN_From_Scratch.ipynb` | Internals with NumPy |
| `Amazon_End_to_End_Pipeline.ipynb` | **Real-world pipeline, start to finish** |
