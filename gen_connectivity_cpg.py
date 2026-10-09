import networkx as nx
import pandas as pd
import useful_ids as uids


def export_subgraph(G: nx.DiGraph,
                    neuron_ids,
                    output_prefix="subgraph",
                    weight_attr="syn_count"):
    """
    Export an induced subgraph for Cytoscape and a connectivity matrix.

    Parameters
    ----------
    G : nx.DiGraph
        Full connectome.
    neuron_ids : iterable
        Neurons to include.
    output_prefix : str
        Prefix for output files.
    weight_attr : str
        Edge attribute containing synapse weight.
        If absent, edges are treated as weight=1.
    """

    neuron_ids = list(neuron_ids)

    # Build induced subgraph
    H = G.subgraph(neuron_ids).copy()

    # ------------------------------------------------------------------
    # Connectivity matrix
    # ------------------------------------------------------------------

    matrix = pd.DataFrame(
        0,
        index=neuron_ids,
        columns=neuron_ids,
        dtype=float,
    )

    for u, v, data in H.edges(data=True):
        matrix.loc[u, v] = data.get(weight_attr, 1)

    matrix.to_csv(f"{output_prefix}_connectivity_matrix.csv")

    # ------------------------------------------------------------------
    # Edge list
    # ------------------------------------------------------------------

    edges = []

    for u, v, data in H.edges(data=True):
        edge = {
            "source": u,
            "target": v,
            "weight": data.get(weight_attr, 1),
        }

        # Preserve any additional edge attributes
        for k, val in data.items():
            if k != weight_attr:
                edge[k] = val

        edges.append(edge)

    pd.DataFrame(edges).to_csv(
        f"{output_prefix}_edge_list.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # GraphML for Cytoscape
    # ------------------------------------------------------------------

    nx.write_graphml(H, f"{output_prefix}.graphml")

    print(f"Nodes : {H.number_of_nodes()}")
    print(f"Edges : {H.number_of_edges()}")
    print(f"Saved:")
    print(f"  {output_prefix}.graphml")
    print(f"  {output_prefix}_connectivity_matrix.csv")
    print(f"  {output_prefix}_edge_list.csv")

    return H, matrix

if __name__ == "__main__":
    import pickle
    banc_graph = pickle.load(open("generated/banc_graph.pkl", "rb"))
    descending_neurons = uids.get_ids(uids.get_descending_neurons("right"))
    print(f"Descending Neurons: {len(descending_neurons):,}")
    cpg_neurons = uids.get_ids(uids.get_cpg_neurons())
    print(f"CPG Neurons: {len(cpg_neurons):,}")
    neurons = descending_neurons + cpg_neurons

    _ = export_subgraph(banc_graph, neurons, output_prefix="generated/descending_cpg_subgraph")
    print("done")