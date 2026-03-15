# Knowledge Graphs, Graph Neural Networks, and Neo4j

A comprehensive, hands-on guide that takes the reader from foundational graph theory through knowledge graph construction in Neo4j to implementing graph neural networks with PyTorch Geometric. The material is grounded in a realistic logistics and supply-chain domain, making the concepts concrete rather than abstract.

## Background and Motivation

This guide grew out of an academic research project I conducted from May through August 2024. The original work investigated the geometry of loss functions in graph neural networks -- specifically, whether the mode connectivity phenomena observed in deep neural networks also hold when the underlying data is graph-structured rather than Euclidean. The research drew on the Fast Geometric Ensembling (FGE) framework proposed by Garipov et al., which demonstrated that the optima of complex loss surfaces in DNNs are connected by simple, low-loss curves, and that traversing these curves enables the construction of high-performing ensembles at the computational cost of training a single model. My contribution was to extend this line of inquiry from standard deep neural networks to GNNs, using the AIDS molecular graph dataset as the experimental testbed.

### What I learned the hard way

The most persistent difficulty I encountered during that research was not any single technical obstacle but rather the disconnect between the theory of knowledge graphs and the theory of graph neural networks. These two fields have developed largely in parallel. Knowledge graph research originates in the semantic web and database communities, where the emphasis is on ontological modeling, entity resolution, and structured query languages such as SPARQL and Cypher. GNN research, by contrast, originates in the machine learning and signal processing communities, where the emphasis is on differentiable message-passing, spectral graph theory, and representation learning. The literature, the tooling, and even the vocabulary differ substantially between the two.

In practice, this meant that learning one did not automatically illuminate the other. I could build a knowledge graph in Neo4j and query it fluently with Cypher, but translating that graph into a format suitable for GNN training -- encoding heterogeneous node types as feature vectors, converting property-graph semantics into a homogeneous adjacency structure, deciding which graph properties to preserve and which to abstract away -- required bridging a gap that no single resource I found at the time addressed clearly. Most tutorials on GNNs used benchmark datasets like Cora or CiteSeer that arrive pre-processed and ready to feed into PyTorch Geometric. Most tutorials on knowledge graphs stopped at querying and visualization. The messy, essential work of moving data from one paradigm to the other was left as an exercise for the reader.

The theoretical side posed its own challenges. GNN papers assume fluency in spectral graph theory, message-passing formalisms, and permutation equivariance -- concepts that are not part of a standard machine learning curriculum. Reading the foundational GCN paper by Kipf and Welling, the GraphSAGE paper by Hamilton et al., or the GAT paper by Velickovic et al. requires comfort with normalized adjacency operators, neighborhood aggregation proofs, and attention mechanisms adapted to irregular topologies. Without a guided path through these ideas, the learning curve is steep and the risk of superficial understanding is high.

### What this guide provides

This notebook is the resource I wished I had when I started. It treats knowledge graphs and graph neural networks as parts of a single pipeline rather than as separate disciplines. Section 5 builds a realistic knowledge graph in Neo4j. Section 8 converts that same graph into PyTorch Geometric tensors and trains GNN models on it. The transition between the two is explicit, documented, and reproducible -- no hand-waving, no pre-processed dataset that hides the complexity.

The theoretical sections (particularly Sections 7 and 8) walk through GCN, GraphSAGE, and GAT at the level of their mathematical formulations, but they do so alongside concrete logistics examples and implementation code. The goal is not to replace the original papers but to make them accessible -- to give you enough grounding that when you do read Kipf and Welling or Hamilton et al., the notation is familiar and the design choices make sense.

If you are a graduate student beginning research in this area, or a practitioner who needs to build a GNN pipeline on top of an existing knowledge graph, this guide is designed to save you the months of disjointed reading and trial-and-error that I went through. The field is moving quickly, but the fundamentals covered here -- graph data modeling, message-passing theory, and the practical bridge between knowledge representation and learned embeddings -- are stable and will remain relevant regardless of which specific architecture is state-of-the-art when you read this.

## Who This Guide Is For

This guide is intended for graduate students, early-career data scientists, and software engineers who are entering the field of graph-based machine learning for the first time. It assumes working knowledge of Python and a basic familiarity with machine learning concepts (loss functions, gradient descent, train/test splits), but it does not assume any prior experience with graph databases, knowledge graphs, or GNNs.

If you can write a Python class and understand what a neural network does at a high level, you have enough background to work through this notebook from start to finish.

## What This Guide Covers

The notebook is organized into ten sections. Each section builds on the previous one, so it is best read sequentially on a first pass. After that, individual sections can serve as standalone references.

| Section | Topic | What You Will Learn |
|---------|-------|---------------------|
| 1 | Introduction and Motivation | Why graph-structured data matters and where knowledge graphs are used in industry |
| 2 | Knowledge Graph Theory | Ontologies, RDF, property graphs, and the formal underpinnings of knowledge representation |
| 3 | Graph Theory Fundamentals | Adjacency matrices, degree distributions, graph types, and hands-on work with NetworkX |
| 4 | Neo4j Setup and Fundamentals | Installing Neo4j, writing Cypher queries, and performing CRUD operations on a graph database |
| 5 | UPS Logistics Knowledge Graph | Designing a schema, generating realistic logistics data, and loading it into Neo4j with parameterized queries |
| 6 | Graph Analytics | Shortest path (Dijkstra), PageRank, community detection, and centrality measures applied to the logistics graph |
| 7 | GNN Theory | The message-passing paradigm, GCN, GraphSAGE, GAT architectures, and the over-smoothing problem |
| 8 | GNN Implementation | Converting a knowledge graph to PyTorch Geometric format, training a GCN for node classification, link prediction, and embedding visualization |
| 9 | Decision Framework | When to use a knowledge graph, when to use a GNN, and when to combine the two |
| 10 | Action Plan and Portfolio Projects | A 30/60/90-day learning roadmap, project ideas, paper reading list, and interview preparation guidance |

## Prerequisites

### Software

- **Python 3.9 or later**
- **Jupyter Notebook or JupyterLab** for running the `.ipynb` file
- **Neo4j Desktop** (free, available at [neo4j.com/download](https://neo4j.com/download/)) with the **Graph Data Science (GDS) plugin** installed. Section 4 of the notebook walks through installation in detail.

### Python Libraries

The notebook installs most dependencies inline, but the core libraries used are:

| Library | Purpose |
|---------|---------|
| `networkx` | Graph construction and classical graph algorithms |
| `neo4j` | Python driver for connecting to a Neo4j database |
| `torch` | PyTorch, the deep learning framework |
| `torch_geometric` | PyTorch Geometric, the GNN library |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | t-SNE and evaluation metrics |
| `numpy`, `pandas` | Numerical computation and data manipulation |

You can install the non-PyG dependencies with:

```bash
pip install networkx neo4j matplotlib seaborn scikit-learn numpy pandas
```

PyTorch and PyTorch Geometric have platform-specific installation steps. The notebook includes cells that handle this, but consult the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if you encounter issues.

### Installing the Neo4j GDS Plugin

Several cells in Section 6 use Neo4j's Graph Data Science procedures (`gds.graph.project`, `gds.shortestPath.dijkstra.stream`, etc.). These are not included in a default Neo4j installation. To install the plugin:

1. Open Neo4j Desktop.
2. Stop your database if it is currently running.
3. Click on your database, then open the **Plugins** tab in the right panel.
4. Locate **Graph Data Science Library** and click **Install**.
5. Restart the database.

You can verify the installation succeeded by running the following Cypher query in the Neo4j Browser:

```cypher
RETURN gds.version()
```

If the GDS plugin is not available, the notebook will still function. All graph analytics are implemented in both Neo4j/Cypher and pure NetworkX, so you can follow along with NetworkX alone.

## How to Navigate This Guide

**If you are completely new to graphs:** Start at Section 1 and work through each section in order. The early sections establish vocabulary and intuition that the later sections depend on. Expect the full notebook to take several sessions to complete.

**If you are comfortable with graph theory but new to Neo4j:** Begin at Section 4. Sections 4 and 5 form a self-contained tutorial on building and querying a knowledge graph in Neo4j.

**If you already use Neo4j and want to learn GNNs:** Skip to Section 7 for the theory and Section 8 for the implementation. Section 8 shows how to convert a knowledge graph into PyTorch Geometric format, which is the bridge between the two worlds.

**If you are preparing for interviews or building a portfolio:** Section 10 contains a structured learning roadmap, portfolio project ideas with architecture diagrams, a curated paper reading list, and specific guidance on how to present graph-based work to technical audiences.

## Repository Structure

```
.
├── Knowledge_Graph_GNN_Neo4j_Guide.ipynb   # The complete guide
├── README.md                                # This file
└── .gitignore
```

## Running the Notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/nhp-atel/Knowledge-Graphs-Graph-Neural-Networks-Neo4j.git
   cd Knowledge-Graphs-Graph-Neural-Networks-Neo4j
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install jupyter networkx neo4j matplotlib seaborn scikit-learn numpy pandas
   ```

4. Launch Jupyter:

   ```bash
   jupyter notebook Knowledge_Graph_GNN_Neo4j_Guide.ipynb
   ```

5. For cells that connect to Neo4j, make sure your Neo4j instance is running and update the connection credentials in the notebook if they differ from the defaults (`bolt://localhost:7687`, user `neo4j`).

## Key Concepts at a Glance

For readers who want a quick orientation before diving into the notebook:

- A **knowledge graph** represents real-world entities as nodes and their relationships as edges, with properties attached to both. Unlike relational tables, it makes the connections between data explicit and queryable.

- **Cypher** is Neo4j's query language. It uses an ASCII-art pattern syntax (e.g., `(a)-[:SHIPS_TO]->(b)`) that visually mirrors the graph structure you are querying.

- **Graph Neural Networks** extend deep learning to irregular, non-Euclidean data. They work by iteratively passing messages between neighboring nodes, allowing each node to build a representation informed by its local graph structure.

- **PyTorch Geometric** is the library used in this guide for GNN implementation. It provides efficient implementations of GCN, GraphSAGE, GAT, and many other architectures.

## License

This project is made available for educational purposes.

## Author

Nimesh Patel
