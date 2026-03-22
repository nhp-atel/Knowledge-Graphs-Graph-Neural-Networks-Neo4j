# Amazon End-to-End Pipeline Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Jupyter notebook that demonstrates the full graph ML pipeline (raw Amazon data → Neo4j → GNN link prediction) using PyTorch Geometric.

**Architecture:** A single Jupyter notebook (`Amazon_End_to_End_Pipeline.ipynb`) with 7 sequential sections (~25-30 cells). Each section is a group of markdown + code cells. Data flows linearly: .mtx files → NetworkX → subsample → Neo4j → PyG Data → GCN model → predictions.

**Tech Stack:** Python 3.9+, scipy, networkx, neo4j, torch, torch_geometric, matplotlib, numpy, pandas, scikit-learn

**Spec:** `docs/superpowers/specs/2026-03-22-amazon-pipeline-notebook-design.md`

---

### Task 1: Create notebook with Section 1 — Introduction & Data Exploration

**Files:**
- Create: `Amazon_End_to_End_Pipeline.ipynb`
- Read: `data/com-Amazon.mtx`, `data/com-Amazon_nodeid.mtx`

- [ ] **Step 1: Create the notebook file with the title markdown cell**

```python
# Cell 0 (markdown):
"""
# Amazon Co-Purchasing Network: End-to-End GNN Pipeline

This notebook takes a real Amazon product co-purchasing dataset and runs the full graph ML pipeline:

**Raw data → Neo4j graph database → PyTorch Geometric → GNN link prediction**

Every step is shown explicitly — no pre-processed datasets, no skipped steps.

---

## About the Dataset

This dataset comes from the [Stanford Network Analysis Platform (SNAP)](https://snap.stanford.edu/data/com-Amazon.html). It captures Amazon's product co-purchasing network:

- **Nodes** are products sold on Amazon
- **Edges** mean "customers who bought product X also bought product Y"
- **Communities** represent product categories (e.g., books, electronics)

The network contains **334,863 products** and **925,872 co-purchase links**.

Our task: **link prediction** — can a GNN learn which products are likely to be co-purchased, based on the structure of the network?

---

## Section 1: Load and Explore the Data
"""
```

- [ ] **Step 2: Add the imports and seeds cell**

```python
# Cell 1 (code):
# ==============================================================
# Imports and Setup
# ==============================================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.io import mmread
from collections import deque
from neo4j import GraphDatabase

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Torch imports (used in later sections)
import torch
torch.manual_seed(SEED)
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

print("All imports loaded successfully.")
```

- [ ] **Step 3: Add the data loading cell**

```python
# Cell 2 (code):
# ==============================================================
# Load the Amazon Co-Purchasing Graph
# ==============================================================

# Load the adjacency matrix (Matrix Market format)
adj_sparse = mmread("data/com-Amazon.mtx")
G_full = nx.from_scipy_sparse_array(adj_sparse)

print(f"Nodes:              {G_full.number_of_nodes():,}")
print(f"Edges:              {G_full.number_of_edges():,}")
print(f"Density:            {nx.density(G_full):.6f}")
print(f"Connected components: {nx.number_connected_components(G_full)}")
```

- [ ] **Step 4: Add the node ID mapping cell with explanatory markdown**

```python
# Cell 3 (markdown):
"""
### Original Amazon Product IDs

The dataset maps each internal node index to an original Amazon product ID.
These are opaque numeric identifiers (not human-readable product names) — the
purpose is to show that a mapping exists back to the original Amazon catalog.
"""

# Cell 4 (code):
# ==============================================================
# Load Original Amazon Product ID Mapping
# ==============================================================

node_ids_raw = mmread("data/com-Amazon_nodeid.mtx")
# Note: this .mtx file is "array" format, so mmread returns a numpy array (not sparse)
node_ids = np.array(node_ids_raw).flatten().astype(int)

print(f"Number of product IDs: {len(node_ids):,}")
print(f"First 10 IDs:  {node_ids[:10]}")
print(f"Last 10 IDs:   {node_ids[-10:]}")

# Build a lookup: internal index -> original Amazon ID
idx_to_amazon_id = {i: int(node_ids[i]) for i in range(len(node_ids))}
print(f"\nExample: internal node 0 -> Amazon product ID {idx_to_amazon_id[0]}")
```

- [ ] **Step 5: Add degree distribution cell**

```python
# Cell 5 (code):
# ==============================================================
# Degree Distribution
# ==============================================================

degrees = [d for _, d in G_full.degree()]

print(f"Min degree:    {min(degrees)}")
print(f"Max degree:    {max(degrees)}")
print(f"Mean degree:   {np.mean(degrees):.1f}")
print(f"Median degree: {np.median(degrees):.1f}")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(degrees, bins=100, edgecolor="black", alpha=0.7, color="#4A90D9")
ax.set_xlabel("Degree", fontsize=12)
ax.set_ylabel("Number of Nodes", fontsize=12)
ax.set_title("Degree Distribution (full graph)", fontsize=14, fontweight="bold")
ax.set_xlim(0, 50)  # Focus on the bulk of the distribution
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
```

- [ ] **Step 6: Add 2-hop neighborhood visualization cell**

```python
# Cell 6 (code):
# ==============================================================
# Visualize a Small Neighborhood
# ==============================================================

# Pick a node with a moderate degree (not too sparse, not a mega-hub)
seed_node = None
for node, deg in G_full.degree():
    if 8 <= deg <= 15:
        seed_node = node
        break

# Extract 2-hop neighborhood
neighbors_1 = set(G_full.neighbors(seed_node))
neighbors_2 = set()
for n in neighbors_1:
    neighbors_2.update(G_full.neighbors(n))

subgraph_nodes = {seed_node} | neighbors_1 | neighbors_2
# Cap at 80 nodes for readability
if len(subgraph_nodes) > 80:
    subgraph_nodes = {seed_node} | neighbors_1 | set(list(neighbors_2)[:80 - len(neighbors_1) - 1])

sub = G_full.subgraph(subgraph_nodes)

fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(sub, seed=SEED, k=0.5)

# Color: seed=red, 1-hop=orange, 2-hop=lightblue
colors = []
for n in sub.nodes():
    if n == seed_node:
        colors.append("#E8734A")
    elif n in neighbors_1:
        colors.append("#F5A623")
    else:
        colors.append("#A8D8EA")

nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.3, edge_color="#888888")
nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=100, edgecolors="white", linewidths=0.5)
ax.set_title(f"2-Hop Neighborhood of Node {seed_node} (Amazon product {idx_to_amazon_id[seed_node]})", fontsize=13, fontweight="bold")
ax.axis("off")

from matplotlib.patches import Patch
legend = [Patch(color="#E8734A", label="Seed node"),
          Patch(color="#F5A623", label="1-hop neighbors"),
          Patch(color="#A8D8EA", label="2-hop neighbors")]
ax.legend(handles=legend, loc="upper left", fontsize=10)
plt.tight_layout()
plt.show()

print(f"Seed node degree: {G_full.degree(seed_node)}")
print(f"Subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
```

- [ ] **Step 7: Verify the notebook runs**

Open the notebook in Jupyter and run all Section 1 cells (Cells 0–6). Verify:
- All imports succeed
- Graph loads with 334,863 nodes, 925,872 edges
- Degree histogram renders
- 2-hop subgraph visualization renders

- [ ] **Step 8: Commit**

```bash
git add Amazon_End_to_End_Pipeline.ipynb
git commit -m "feat: add Section 1 — introduction and data exploration"
```

---

### Task 2: Add Section 2 — Subsampling

**Files:**
- Modify: `Amazon_End_to_End_Pipeline.ipynb`
- Read: `data/com-Amazon_Communities_all.mtx`

- [ ] **Step 1: Add subsampling markdown cell**

```python
# Cell 7 (markdown):
"""
---

## Section 2: Subsample the Graph

The full graph has 334K nodes — too large for a quick tutorial. We extract a connected subgraph of ~10,000 nodes using BFS from a fixed seed node. This preserves the local community structure (unlike random node sampling, which fragments the graph into disconnected pieces).
"""
```

- [ ] **Step 2: Add BFS subsampling + community filtering code cell**

```python
# Cell 8 (code):
# ==============================================================
# BFS Subsampling
# ==============================================================

TARGET_SIZE = 10000
BFS_SEED = 0  # Fixed seed node for reproducibility

def bfs_subsample(G, seed, target_size):
    """Extract a connected subgraph via BFS from a seed node."""
    visited = set()
    queue = deque([seed])
    visited.add(seed)

    while queue and len(visited) < target_size:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= target_size:
                    break
    return visited

sampled_nodes = bfs_subsample(G_full, BFS_SEED, TARGET_SIZE)
G_sub = G_full.subgraph(sampled_nodes).copy()

# Build mapping: original node ID -> new index (0..N-1)
orig_to_new = {orig: new for new, orig in enumerate(sorted(G_sub.nodes()))}
G_sub = nx.relabel_nodes(G_sub, orig_to_new)
sampled_orig_nodes = sorted(sampled_nodes)  # Keep for community filtering

print(f"Subsampled graph:")
print(f"  Nodes: {G_sub.number_of_nodes():,}")
print(f"  Edges: {G_sub.number_of_edges():,}")
print(f"  Density: {nx.density(G_sub):.6f}")
print(f"  Connected: {nx.is_connected(G_sub)}")

print(f"\nOriginal graph:")
print(f"  Nodes: {G_full.number_of_nodes():,}")
print(f"  Edges: {G_full.number_of_edges():,}")
print(f"  Density: {nx.density(G_full):.6f}")
```

- [ ] **Step 3: Add community matrix filtering cell**

```python
# Cell 9 (code):
# ==============================================================
# Filter Community Assignments to Subsampled Nodes
# ==============================================================

communities_sparse = mmread("data/com-Amazon_Communities_all.mtx").tocsr()
print(f"Full community matrix: {communities_sparse.shape[0]:,} nodes x {communities_sparse.shape[1]:,} communities")

# Slice rows to only our sampled nodes (using original indices)
comm_sub = communities_sparse[sampled_orig_nodes, :]

# Drop communities with fewer than 2 members in the subsample
col_counts = np.array((comm_sub > 0).sum(axis=0)).flatten()
keep_cols = np.where(col_counts >= 2)[0]
comm_sub = comm_sub[:, keep_cols]

# Save per-column member counts AFTER filtering (used later in t-SNE visualization)
comm_col_counts = np.array((comm_sub > 0).sum(axis=0)).flatten()

print(f"Filtered community matrix: {comm_sub.shape[0]:,} nodes x {comm_sub.shape[1]} communities")
print(f"  Communities with 2+ members in subsample: {len(keep_cols)}")

nodes_with_community = (comm_sub.sum(axis=1) > 0).sum()
print(f"  Nodes with at least one community: {int(nodes_with_community):,} / {comm_sub.shape[0]:,}")
```

- [ ] **Step 4: Verify subsampling runs**

Run Cells 7–9. Verify:
- Subsampled graph has ~10,000 nodes, is connected
- Community matrix is filtered to a compact shape
- Most nodes have at least one community assignment

- [ ] **Step 5: Commit**

```bash
git add Amazon_End_to_End_Pipeline.ipynb
git commit -m "feat: add Section 2 — BFS subsampling and community filtering"
```

---

### Task 3: Add Section 3 — Neo4j Ingestion & Querying

**Files:**
- Modify: `Amazon_End_to_End_Pipeline.ipynb`

- [ ] **Step 1: Add Neo4j setup markdown cell**

```python
# Cell 10 (markdown):
"""
---

## Section 3: Load into Neo4j

Now we load the subsampled graph into Neo4j — the graph database step of the pipeline.

**If you already set up Neo4j in the main guide**, you're good to go — just make sure your database is running.

**If you skipped it**, that's completely fine. Here's what you need:

1. Download [Neo4j Desktop](https://neo4j.com/download/) (free)
2. Create a new project and add a local database
3. Set a password (the code below uses `neo4j` / `password` — update if yours differs)
4. Start the database

Once it's running, the cells below will connect, load the data, and let you explore it with Cypher queries.
"""
```

- [ ] **Step 2: Add Neo4j connection cell**

```python
# Cell 11 (code):
# ==============================================================
# Connect to Neo4j
# ==============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Update this to match your database password

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Verify connection
with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print(f"Neo4j connection successful: {result.single()['test'] == 1}")
```

- [ ] **Step 3: Add batch ingestion cell**

```python
# Cell 12 (code):
# ==============================================================
# Batch Insert Products and Co-Purchase Edges
# ==============================================================

def clear_database(session):
    session.run("MATCH (n) DETACH DELETE n")

def insert_nodes_batch(session, nodes):
    """Insert product nodes using UNWIND for speed."""
    session.run(
        "UNWIND $batch AS row "
        "CREATE (:Product {node_idx: row.node_idx, product_id: row.product_id})",
        batch=nodes
    )

def insert_edges_batch(session, edges, batch_size=5000):
    """Insert co-purchase edges in batches using UNWIND."""
    for i in range(0, len(edges), batch_size):
        batch = edges[i:i + batch_size]
        session.run(
            "UNWIND $batch AS row "
            "MATCH (a:Product {node_idx: row.src}) "
            "MATCH (b:Product {node_idx: row.dst}) "
            "CREATE (a)-[:CO_PURCHASED]->(b)",
            batch=batch
        )

# Prepare node and edge data
node_data = [{"node_idx": int(n), "product_id": int(idx_to_amazon_id[sampled_orig_nodes[n]])}
             for n in G_sub.nodes()]

edge_data = [{"src": int(u), "dst": int(v)} for u, v in G_sub.edges()]

# Insert into Neo4j
with driver.session() as session:
    clear_database(session)
    print("Database cleared.")

    insert_nodes_batch(session, node_data)
    print(f"Inserted {len(node_data):,} product nodes.")

    insert_edges_batch(session, edge_data)
    print(f"Inserted {len(edge_data):,} co-purchase edges.")
```

- [ ] **Step 4: Add Cypher exploration cells**

```python
# Cell 13 (markdown):
"""
### Explore the Graph in Neo4j

Now the data lives in a graph database. Let's run a few Cypher queries to see what's in there.
"""

# Cell 14 (code):
# ==============================================================
# Cypher Queries: Explore the Co-Purchasing Network
# ==============================================================

with driver.session() as session:
    # 1. Verify counts
    node_count = session.run("MATCH (p:Product) RETURN count(p) AS n").single()["n"]
    edge_count = session.run("MATCH ()-[:CO_PURCHASED]->() RETURN count(*) AS n").single()["n"]
    print(f"Products in Neo4j:      {node_count:,}")
    print(f"Co-purchase edges:      {edge_count:,}")

    # 2. Most connected product
    result = session.run("""
        MATCH (p:Product)-[:CO_PURCHASED]-()
        RETURN p.product_id AS product, count(*) AS connections
        ORDER BY connections DESC LIMIT 5
    """)
    print("\nMost connected products:")
    for record in result:
        print(f"  Product {record['product']}: {record['connections']} connections")

    # 3. Co-purchase neighborhood
    top_product = session.run("""
        MATCH (p:Product)-[:CO_PURCHASED]-()
        RETURN p.node_idx AS idx
        ORDER BY count(*) DESC LIMIT 1
    """).single()["idx"]

    result = session.run("""
        MATCH (p:Product {node_idx: $idx})-[:CO_PURCHASED]-(neighbor:Product)
        RETURN neighbor.product_id AS product_id
        LIMIT 10
    """, idx=top_product)
    print(f"\nCustomers who bought product {idx_to_amazon_id[sampled_orig_nodes[top_product]]} also bought:")
    for record in result:
        print(f"  Product {record['product_id']}")
```

- [ ] **Step 5: Verify Neo4j section runs**

Run Cells 10–14. Verify:
- Connection succeeds
- Node/edge counts match the subsampled graph
- Cypher queries return results

- [ ] **Step 6: Commit**

```bash
git add Amazon_End_to_End_Pipeline.ipynb
git commit -m "feat: add Section 3 — Neo4j ingestion and Cypher exploration"
```

---

### Task 4: Add Section 4 — Neo4j to PyG Conversion

**Files:**
- Modify: `Amazon_End_to_End_Pipeline.ipynb`

- [ ] **Step 1: Add conversion markdown cell**

```python
# Cell 15 (markdown):
"""
---

## Section 4: From Neo4j to PyTorch Geometric

This is the step most tutorials skip.

The data is sitting in Neo4j — how do we turn it into something a neural network can consume? We need two things: a **feature matrix** (numbers describing each node) and an **edge list** (which nodes connect).

Since this dataset has no product attributes (no names, prices, or descriptions), we engineer features from the graph structure itself. This is realistic — in many real pipelines, your initial features come from the graph topology.
"""
```

- [ ] **Step 2: Add data extraction from Neo4j cell**

```python
# Cell 16 (code):
# ==============================================================
# Pull Data from Neo4j
# ==============================================================

with driver.session() as session:
    # Get all node indices
    result = session.run("MATCH (p:Product) RETURN p.node_idx AS idx ORDER BY idx")
    neo4j_nodes = [record["idx"] for record in result]

    # Get all edges
    result = session.run("""
        MATCH (a:Product)-[:CO_PURCHASED]->(b:Product)
        RETURN a.node_idx AS src, b.node_idx AS dst
    """)
    neo4j_edges = [(record["src"], record["dst"]) for record in result]

print(f"Pulled from Neo4j: {len(neo4j_nodes):,} nodes, {len(neo4j_edges):,} edges")
```

- [ ] **Step 3: Add feature engineering cell**

```python
# Cell 17 (markdown):
"""
### Feature Engineering from Graph Structure

We build four types of features for each node:

| Feature | Description | Dimensions |
|---------|-------------|------------|
| Degree (normalized) | How many co-purchase links this product has | 1 |
| Clustering coefficient | How interconnected the product's neighbors are | 1 |
| Community count | How many product categories this product belongs to | 1 |
| Community indicators | Which specific communities this product belongs to | ~20-50 |

The community indicator vector is filtered to only include communities that have at least 2 members in our subsample. This keeps the feature matrix compact instead of a sparse 75,000-column vector.
"""

# Cell 18 (code):
# ==============================================================
# Engineer Node Features
# ==============================================================

N = G_sub.number_of_nodes()

# 1. Normalized degree
degrees_sub = np.array([G_sub.degree(n) for n in range(N)], dtype=np.float32)
max_deg = degrees_sub.max()
norm_degree = degrees_sub / max_deg

# 2. Clustering coefficient
clustering = np.array([nx.clustering(G_sub, n) for n in range(N)], dtype=np.float32)

# 3. Community count (from filtered community matrix)
comm_dense = np.array(comm_sub.todense(), dtype=np.float32)
comm_count = comm_dense.sum(axis=1)

# 4. Community indicator vector (already filtered in Section 2)
# comm_dense is already [N, num_filtered_communities]

# Stack all features
features = np.column_stack([
    norm_degree.reshape(-1, 1),
    clustering.reshape(-1, 1),
    comm_count.reshape(-1, 1),
    comm_dense
])

print(f"Feature matrix shape: {features.shape}")
print(f"  - 1 normalized degree")
print(f"  - 1 clustering coefficient")
print(f"  - 1 community count")
print(f"  - {comm_dense.shape[1]} community indicators")
print(f"  - Total: {features.shape[1]} features per node")
```

- [ ] **Step 4: Add PyG Data construction cell**

```python
# Cell 19 (code):
# ==============================================================
# Build PyG Data Object
# ==============================================================

# Node features
x = torch.tensor(features, dtype=torch.float)

# Edge index: COO format [2, num_edges], both directions for undirected
src = [e[0] for e in neo4j_edges]
dst = [e[1] for e in neo4j_edges]
# Add reverse edges for undirected graph
edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

pyg_data = Data(x=x, edge_index=edge_index)

print(pyg_data)
print(f"\nThis is what the GNN will see:")
print(f"  Node features: {pyg_data.x.shape[0]:,} nodes x {pyg_data.x.shape[1]} features")
print(f"  Edges:          {pyg_data.edge_index.shape[1]:,} (both directions)")
```

- [ ] **Step 5: Verify conversion runs**

Run Cells 15–19. Verify:
- Features are extracted from Neo4j
- Feature matrix has expected shape (~10K x ~25-55)
- PyG Data object prints correctly

- [ ] **Step 6: Commit**

```bash
git add Amazon_End_to_End_Pipeline.ipynb
git commit -m "feat: add Section 4 — Neo4j to PyG conversion with structural features"
```

---

### Task 5: Add Section 5 — Train/Test Split

**Files:**
- Modify: `Amazon_End_to_End_Pipeline.ipynb`

- [ ] **Step 1: Add link split markdown and code cells**

```python
# Cell 20 (markdown):
"""
---

## Section 5: Train/Test Split for Link Prediction

To train a link predictor, we need to:

1. **Hide some real edges** — these become our positive test examples
2. **Generate fake edges** (pairs of nodes that are NOT connected) — these become negative examples
3. **Train the GNN** to tell real edges from fake ones

PyG's `RandomLinkSplit` handles all of this automatically.
"""

# Cell 21 (code):
# ==============================================================
# Split Edges for Link Prediction
# ==============================================================

splitter = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
)

train_data, val_data, test_data = splitter(pyg_data)

print(f"Training edges:   {train_data.edge_label_index.shape[1]:,} ({train_data.edge_label.sum().int():,} positive)")
print(f"Validation edges: {val_data.edge_label_index.shape[1]:,} ({val_data.edge_label.sum().int():,} positive)")
print(f"Test edges:       {test_data.edge_label_index.shape[1]:,} ({test_data.edge_label.sum().int():,} positive)")

# Cell 22 (code):
# ==============================================================
# Sanity Check: No Test Edge Leakage
# ==============================================================

train_edges = set(map(tuple, train_data.edge_index.t().tolist()))
test_pos_edges = set(map(tuple, test_data.edge_label_index[:, test_data.edge_label == 1].t().tolist()))

leak = train_edges & test_pos_edges
print(f"Test edges leaked into training graph: {len(leak)}")
assert len(leak) == 0, "LEAK DETECTED — test edges found in training graph!"
print("No leakage detected. Train/test split is clean.")
```

- [ ] **Step 2: Verify split runs**

Run Cells 20–22. Verify:
- Edge counts are printed
- No leakage assertion passes

- [ ] **Step 3: Commit**

```bash
git add Amazon_End_to_End_Pipeline.ipynb
git commit -m "feat: add Section 5 — train/test split for link prediction"
```

---

### Task 6: Add Section 6 — GNN Model & Training

**Files:**
- Modify: `Amazon_End_to_End_Pipeline.ipynb`

- [ ] **Step 1: Add model markdown and definition cell**

```python
# Cell 23 (markdown):
"""
---

## Section 6: Train the GNN

We train a 2-layer Graph Convolutional Network (GCN) for link prediction. The model:

1. **Encodes** each product into a 32-dimensional embedding by passing information through the graph
2. **Decodes** by taking the dot product of two product embeddings — high score means "likely co-purchased"

For the theory behind GCNs and message passing, see Section 7 of the main guide (`Knowledge_Graph_GNN_Neo4j_Guide.ipynb`).
"""

# Cell 24 (code):
# ==============================================================
# Link Prediction GCN Model
# ==============================================================

class LinkPredGCN(torch.nn.Module):
    """2-layer GCN encoder with dot-product decoder for link prediction."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        """Produce node embeddings."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """Dot-product decoder: score = z_u . z_v"""
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


model = LinkPredGCN(in_channels=pyg_data.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
```

- [ ] **Step 2: Add training loop cell**

```python
# Cell 25 (code):
# ==============================================================
# Training Loop
# ==============================================================

NUM_EPOCHS = 200
LOG_EVERY = 10

train_losses = []
val_aucs = []

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    optimizer.zero_grad()
    scores = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(scores, train_data.edge_label)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_scores = model(val_data.x, val_data.edge_index, val_data.edge_label_index)
        val_auc = roc_auc_score(val_data.edge_label.numpy(), val_scores.sigmoid().numpy())
        val_aucs.append(val_auc)

    if epoch % LOG_EVERY == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

print(f"\nBest validation AUC: {max(val_aucs):.4f} (epoch {np.argmax(val_aucs) + 1})")
```

- [ ] **Step 3: Add training curve plot cell**

```python
# Cell 26 (code):
# ==============================================================
# Training Curves
# ==============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, color="#4A90D9", linewidth=1.5)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.plot(val_aucs, color="#E8734A", linewidth=1.5)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("AUC", fontsize=12)
ax2.set_title("Validation AUC", fontsize=14, fontweight="bold")
ax2.set_ylim(0.5, 1.0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Add test evaluation cell**

```python
# Cell 27 (code):
# ==============================================================
# Test Set Evaluation
# ==============================================================

model.eval()
with torch.no_grad():
    test_scores = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    test_auc = roc_auc_score(test_data.edge_label.numpy(), test_scores.sigmoid().numpy())

print(f"Test AUC: {test_auc:.4f}")
```

- [ ] **Step 5: Verify training runs**

Run Cells 23–27. Verify:
- Model prints architecture
- Training loop runs 200 epochs
- Loss decreases, AUC increases
- Training curves render
- Test AUC is printed

- [ ] **Step 6: Commit**

```bash
git add Amazon_End_to_End_Pipeline.ipynb
git commit -m "feat: add Section 6 — GCN model and training loop"
```

---

### Task 7: Add Section 7 — Results & Visualization

**Files:**
- Modify: `Amazon_End_to_End_Pipeline.ipynb`

- [ ] **Step 1: Add results markdown cell**

```python
# Cell 28 (markdown):
"""
---

## Section 7: What Did the Model Learn?

Let's visualize the learned embeddings and see the model's top predictions for new co-purchase links.
"""
```

- [ ] **Step 2: Add t-SNE visualization cell**

```python
# Cell 29 (code):
# ==============================================================
# t-SNE Visualization of Learned Embeddings
# ==============================================================

model.eval()
with torch.no_grad():
    embeddings = model.encode(pyg_data.x, train_data.edge_index).numpy()

# Assign dominant community (largest community the node belongs to)
# comm_sub and comm_col_counts were defined in Section 2 (Cell 9)
comm_array = np.array(comm_sub.todense())
dominant_community = np.full(N, -1)  # -1 = uncategorized
for i in range(N):
    memberships = np.where(comm_array[i] > 0)[0]
    if len(memberships) > 0:
        # Pick the community with the most members in the subsample
        sizes = [comm_col_counts[c] for c in memberships]
        dominant_community[i] = memberships[np.argmax(sizes)]

# t-SNE reduction
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
coords = tsne.fit_transform(embeddings)

# Plot
fig, ax = plt.subplots(figsize=(12, 9))

# Uncategorized nodes in light gray
uncategorized = dominant_community == -1
ax.scatter(coords[uncategorized, 0], coords[uncategorized, 1],
           c="#CCCCCC", s=5, alpha=0.3, label="Uncategorized")

# Categorized nodes colored by community
categorized = ~uncategorized
scatter = ax.scatter(coords[categorized, 0], coords[categorized, 1],
                     c=dominant_community[categorized], cmap="tab20",
                     s=8, alpha=0.5)

ax.set_title("t-SNE of Learned Product Embeddings\n(colored by dominant product community)", fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.show()

n_cat = int(categorized.sum())
print(f"Categorized: {n_cat:,} / {N:,} nodes ({100 * n_cat / N:.1f}%)")
```

- [ ] **Step 3: Add top predicted links cell**

```python
# Cell 30 (code):
# ==============================================================
# Top 10 Predicted New Co-Purchase Links
# ==============================================================

model.eval()
with torch.no_grad():
    z = model.encode(pyg_data.x, train_data.edge_index)

# Get existing edges as a set for filtering
existing_edges = set(map(tuple, pyg_data.edge_index.t().tolist()))

# Sample candidate non-edges and score them
candidates = []
tested = set()
np.random.seed(SEED)
while len(candidates) < 50000:
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges and (i, j) not in tested:
        tested.add((i, j))
        score = (z[i] * z[j]).sum().item()
        candidates.append((i, j, score))

# Sort by score descending
candidates.sort(key=lambda x: x[2], reverse=True)

# Display top 10
print("Top 10 Predicted New Co-Purchase Links:\n")
print(f"{'Rank':<6} {'Node A':<10} {'Node B':<10} {'Amazon ID A':<15} {'Amazon ID B':<15} {'Score':<10}")
print("-" * 66)
for rank, (i, j, score) in enumerate(candidates[:10], 1):
    aid_a = idx_to_amazon_id[sampled_orig_nodes[i]]
    aid_b = idx_to_amazon_id[sampled_orig_nodes[j]]
    print(f"{rank:<6} {i:<10} {j:<10} {aid_a:<15} {aid_b:<15} {score:<10.4f}")
```

- [ ] **Step 4: Add closing markdown cell**

```python
# Cell 31 (markdown):
"""
---

## Summary

This notebook ran the full pipeline on real data:

1. **Loaded** the Amazon co-purchasing network (334K products, 925K edges)
2. **Explored** its structure — degree distribution, neighborhood visualization
3. **Subsampled** a 10K-node connected subgraph via BFS
4. **Ingested** it into Neo4j and explored it with Cypher queries
5. **Converted** it from Neo4j into PyTorch Geometric format — the step most tutorials skip
6. **Trained** a GCN to predict co-purchase links
7. **Visualized** the learned embeddings and predicted new connections

The same pipeline works for any graph database: swap the data source, adjust the feature engineering, and the rest stays the same.

For the theory behind GNNs, see the main guide (`Knowledge_Graph_GNN_Neo4j_Guide.ipynb`). For a from-scratch NumPy implementation, see `GNN_From_Scratch.ipynb`.
"""
```

- [ ] **Step 5: Verify results section runs**

Run Cells 28–31. Verify:
- t-SNE plot renders with colored clusters
- Top 10 predictions table prints with Amazon product IDs
- No errors

- [ ] **Step 6: Commit**

```bash
git add Amazon_End_to_End_Pipeline.ipynb
git commit -m "feat: add Section 7 — t-SNE visualization and predicted links"
```

---

### Task 8: Update README and Final Cleanup

**Files:**
- Modify: `README.md` (repository structure section, "Which Notebook" section)
- Modify: `Amazon_End_to_End_Pipeline.ipynb` (final review)

- [ ] **Step 1: Update README "Which Notebook Should I Start With?" section**

Add the new notebook to the table:

```markdown
| `Amazon_End_to_End_Pipeline.ipynb` | scipy, NetworkX, Neo4j, PyTorch, PyTorch Geometric | Full pipeline on real Amazon data: raw .mtx → Neo4j → GNN link prediction |
```

Add guidance text:

```markdown
**Then try the Amazon pipeline notebook** when you want to see the full process on a real dataset. It takes the Amazon co-purchasing network, loads it into Neo4j, converts it to PyTorch Geometric format, and trains a GNN to predict which products will be co-purchased — end to end, nothing skipped.
```

- [ ] **Step 2: Update README "Repository Structure" section**

Add the new notebook and data directory:

```
├── Amazon_End_to_End_Pipeline.ipynb    # Real-world pipeline (Amazon data → Neo4j → GNN)
├── data/                               # Amazon co-purchasing dataset (SNAP)
```

- [ ] **Step 3: Run full notebook end to end**

Restart kernel and run all cells. Verify every section completes without errors.

- [ ] **Step 4: Commit**

```bash
git add README.md Amazon_End_to_End_Pipeline.ipynb
git commit -m "docs: update README with Amazon pipeline notebook"
```
