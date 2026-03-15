# Knowledge Graphs, Graph Neural Networks, and Neo4j

A hands-on guide that walks you from "what is a graph?" all the way through building a graph database and training a neural network on it. Everything is grounded in a real-world logistics example, so the concepts stay concrete.

## The Short Version

Most data we work with lives in tables -- rows and columns. But some data is better described by **connections**. Think about a shipping network: warehouses connect to drivers, drivers carry packages, packages travel along routes between cities. That web of connections is a **graph**, and there are powerful tools designed specifically to store, query, and learn from it.

This guide covers three of those tools and how they fit together:

```
  Knowledge Graph          Graph Database            Graph Neural Network
  (the idea)               (the storage)             (the ML model)

  "Warehouses connect      Store it in Neo4j,        Feed the graph into a
   to drivers, drivers     query it with Cypher:     neural network that
   carry packages,         MATCH (w)-[:EMPLOYS]->    learns from the shape
   packages travel         (d) RETURN w, d           of the connections
   along routes"                                     themselves
```

If you have worked with SQL databases or trained a neural network on tabular data, you already have enough background. This guide handles the rest.

## Background and Motivation

This guide grew out of an academic research project I conducted from May through August 2024. The research question was straightforward: when you train a neural network, the optimization process finds a set of parameters (a "solution") that performs well. It turns out that different training runs can find different solutions, and recent work by Garipov et al. showed that in standard deep neural networks, these solutions are connected by simple, smooth paths -- you can walk from one solution to another without the model's performance collapsing. My project asked whether the same property holds when the neural network operates on graph-structured data (molecular graphs from the AIDS dataset) instead of the usual images or tabular inputs.

That research required me to work at the intersection of two fields: **knowledge graphs** (how you structure and store connected data) and **graph neural networks** (how you train ML models on that data). What I found is that these two fields have grown up in different communities, use different vocabulary, and are taught in almost entirely separate resources.

### The gap I kept running into

Learning knowledge graphs and learning GNNs felt like studying two different subjects that happen to share the word "graph."

On the knowledge graph side, tutorials teach you how to model entities and relationships, store them in a database like Neo4j, and write queries to retrieve information. They stop there. The data sits in the database, ready to be queried, but not ready for machine learning.

On the GNN side, tutorials hand you a pre-packaged dataset that is already formatted as numerical arrays -- node features, edge lists, labels -- and show you how to train a model on it. They skip the question of where that data came from or how a real-world graph database gets converted into something a neural network can consume.

The critical middle step -- taking a knowledge graph out of Neo4j and turning it into the numerical format a GNN expects -- was the part I struggled with most, and the part that no single resource explained clearly. That meant figuring out questions like:

- How do you turn a node that has a label like "warehouse" and properties like "capacity: 5000" into a row of numbers?
- If your graph has different types of nodes (warehouses, packages, drivers), how do you represent that in a single feature matrix?
- Which relationships from the knowledge graph become edges in the GNN, and which become features?

### What this guide does differently

This notebook treats the full pipeline as one continuous workflow. Section 5 builds a logistics knowledge graph in Neo4j from scratch. Section 8 takes that same graph, converts it into numerical tensors, and trains GNN models on it. Every step in between is shown explicitly -- no pre-processed datasets, no skipped steps.

The theory sections explain GNN architectures (GCN, GraphSAGE, GAT) with their mathematical formulations, but they pair every equation with a plain-language explanation of what it does in the context of the logistics network. The goal is to give you enough grounding that when you encounter these concepts in research papers, they feel familiar rather than foreign.

If you are starting out in this area, this guide is designed to save you the months of scattered reading and dead ends that I went through.

## Who This Guide Is For

This guide is written for graduate students, early-career data scientists, and software engineers who are new to graph-based machine learning. It assumes you can write Python and that you have a basic understanding of how neural networks work (what a loss function is, what training means), but it does not assume any prior experience with graph databases, knowledge graphs, or GNNs.

If you can write a Python class and understand what a neural network does at a high level, you have enough background to work through this notebook.

## What This Guide Covers

The notebook is organized into ten sections. Each section builds on the previous one, so it is best read sequentially on a first pass. After that, individual sections can serve as standalone references.

| Section | Topic | What You Will Learn |
|---------|-------|---------------------|
| 1 | Introduction and Motivation | Why some data is better represented as a graph than a table, and where this matters in industry |
| 2 | Knowledge Graph Theory | How to formally describe entities and their relationships -- the different standards and formats used |
| 3 | Graph Theory Fundamentals | The math behind graphs: how to represent them as matrices, measure their properties, and work with them in Python (NetworkX) |
| 4 | Neo4j Setup and Fundamentals | Installing Neo4j (a graph database), writing queries in its language (Cypher), and creating/reading/updating/deleting data |
| 5 | UPS Logistics Knowledge Graph | Building a realistic shipping network from scratch -- warehouses, drivers, packages, and routes -- and loading it into Neo4j |
| 6 | Graph Analytics | Algorithms that answer questions like "what is the shortest route?" and "which warehouse is the most critical hub?" |
| 7 | GNN Theory | How graph neural networks work -- each node learns by collecting information from its neighbors, layer by layer |
| 8 | GNN Implementation | The full pipeline: taking the logistics graph from Neo4j, converting it to numerical format, and training a neural network on it |
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

## Which Notebook Should I Start With?

This repository contains two notebooks. They cover overlapping material but serve different purposes.

| Notebook | What It Uses | What It Is For |
|----------|-------------|----------------|
| `Knowledge_Graph_GNN_Neo4j_Guide.ipynb` | NetworkX, Neo4j, PyTorch, PyTorch Geometric | The full guide: theory, tools, and implementation using standard libraries |
| `GNN_From_Scratch.ipynb` | NumPy only (no PyTorch, no PyG) | Shows how GNNs work under the hood by implementing every operation by hand |

**Start with the main guide.** It covers the theory, walks through Neo4j and Cypher, and builds up to GNN implementation using the tools you would actually use in production.

**Then read the from-scratch notebook** when you want to understand what those tools are doing internally. It implements the same GCN architecture, the same training loop, the same link prediction pipeline -- but every weight matrix, every gradient, every optimizer update is written out explicitly in NumPy. There is no `loss.backward()`, no `GCNConv`. You see the raw matrix multiplications that make a graph neural network work.

## Repository Structure

```
.
├── Knowledge_Graph_GNN_Neo4j_Guide.ipynb   # The complete guide (theory + standard libraries)
├── GNN_From_Scratch.ipynb                  # GNN from scratch (NumPy only, no PyTorch)
├── README.md                                # This file
└── .gitignore
```

## Running the Notebooks

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

If these terms are new to you, here is what they mean in plain language:

- A **graph** is a collection of things (called **nodes**) connected by relationships (called **edges**). A social network is a graph: people are nodes, friendships are edges. A shipping network is a graph: warehouses and packages are nodes, routes and deliveries are edges.

- A **knowledge graph** is a graph where the nodes and edges carry meaningful labels and properties. It is not just "A connects to B" -- it is "Warehouse-Seattle SHIPS_TO Warehouse-Chicago, distance: 1750 miles." It is a way of storing real-world knowledge so that both humans and machines can query it.

- **Neo4j** is a database built specifically for storing and querying graphs. Instead of SQL, it uses a language called **Cypher** that looks like ASCII art: `(seattle)-[:SHIPS_TO]->(chicago)`. You draw the pattern you are looking for, and Neo4j finds it in the data.

- A **Graph Neural Network (GNN)** is a neural network designed to learn from graph-structured data. The core idea is simple: each node builds its understanding by collecting information from its neighbors. After several rounds of this, every node has learned something about the broader network around it -- and you can use that learned representation for predictions.

- **PyTorch Geometric** is the Python library used in this guide to build and train GNNs. It is to graph neural networks what PyTorch is to standard neural networks -- it handles the data formatting, the model layers, and the training loop.

## License

This project is made available for educational purposes.

## Author

Nimesh Patel
