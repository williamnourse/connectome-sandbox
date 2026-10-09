#!/usr/bin/env python3

import pickle

import networkx as nx
import polars as pl

from tqdm import tqdm

# =============================================================================
# Settings
# =============================================================================

CONNECTIVITY_FILE = "data/banc_connections_v888.parquet"
NEURONS_FILE = "data/banc_neurons_v888.parquet"
OUTPUT_GRAPH = "generated/banc_graph"

MIN_SYNAPSES = 3

# =============================================================================
# Load connectivity
# =============================================================================

print("Loading connectivity table...")

connectivity = pl.read_parquet(CONNECTIVITY_FILE)
neurons = pl.read_parquet(NEURONS_FILE)

edges = connectivity

# =============================================================================
# Filter weak connections
# =============================================================================

edges = edges.filter(
    pl.col("syn_count") >= MIN_SYNAPSES
)

print(f"Remaining connections: {len(edges):,}")

# =============================================================================
# Build directed graph
# =============================================================================

print("Building directed graph...")

G = nx.from_pandas_edgelist(
    edges.to_pandas(),
    source="pre_root_id",
    target="post_root_id",
    edge_attr=True,
    create_using=nx.DiGraph,
)

# =============================================================================
# Attach all neuron attributes
# =============================================================================

print("Adding neuron metadata...")

metadata = (
    neurons
    .select([
        "Root ID",        # or "root_id" depending on your table
        "Primary Cell Type",      # or "primary_cell_type" depending on your table
        "Predicted NT type",   # adjust name if necessary
    ])
    .to_dicts()
)

for row in tqdm(metadata):

    root_id = row["Root ID"]

    if root_id not in G:
        continue

    G.nodes[root_id]["cell_type"] = row["Primary Cell Type"] or "Unknown"
    G.nodes[root_id]["predicted_nt"] = row["Predicted NT type"] or "Unknown"

# =============================================================================
# Summary
# =============================================================================

print(f"Nodes : {G.number_of_nodes():,}")
print(f"Edges : {G.number_of_edges():,}")

# =============================================================================
# Save graph
# =============================================================================

print(f"Saving graph to {OUTPUT_GRAPH}")

with open(OUTPUT_GRAPH+'.pkl', "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saving graph to {OUTPUT_GRAPH} in GraphML format")

nx.write_graphml(G, OUTPUT_GRAPH+'.graphml')

print("Done.")