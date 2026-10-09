from collections import deque

import networkx as nx
import useful_ids as uids
from tqdm import tqdm

def multi_source_shortest_path_length(G, sources):
    """
    Compute shortest path lengths from the nearest source
    to every reachable node in an unweighted graph.
    """
    dist = {node: 0 for node in sources}
    queue = deque(sources)

    while queue:
        u = queue.popleft()

        for v in G.successors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist

def intermediate_subgraph(
    G: nx.DiGraph,
    A,
    B,
    C=None,
    max_total_steps=None,
):
    """
    Return the induced subgraph containing neurons that are:
      1. downstream of any neuron in A,
      2. upstream of any neuron in B,
      3. optionally in C,
      4. optionally on an A→B shortest path whose total length is
         <= max_total_steps.

    Parameters
    ----------
    G : nx.DiGraph
        Directed connectome graph (presynaptic -> postsynaptic).
    A : iterable
        Source neuron IDs.
    B : iterable
        Target neuron IDs.
    C : iterable, optional
        Restrict results to this set of neurons.
    max_total_steps : int, optional
        Maximum allowed shortest-path length from A to B through a node.
    """

    # Distance from the nearest source in A
    dist_from_A = multi_source_shortest_path_length(G, A)

    # Distance to the nearest target in B
    G_rev = G.reverse(copy=False)
    dist_to_B = multi_source_shortest_path_length(G_rev, B)

    # Nodes lying on at least one directed path from A to B
    keep = set(dist_from_A) & set(dist_to_B)

    # Restrict by total path length
    if max_total_steps is not None:
        keep = {
            node
            for node in keep
            if dist_from_A[node] + dist_to_B[node] <= max_total_steps
        }

    # Optional whitelist
    if C is not None:
        keep &= set(C)

    return G.subgraph(keep).copy()

if __name__ == "__main__":
    import pickle
    banc_graph = pickle.load(open("generated/banc_graph.pkl", "rb"))
    descending_neurons = uids.get_ids(uids.get_descending_neurons("right"))
    print(f"Descending Neurons: {len(descending_neurons):,}")
    cpg_neurons = uids.get_ids(uids.get_cpg_neurons())
    print(f"CPG Neurons: {len(cpg_neurons):,}")
    motor_neurons = uids.get_ids(uids.get_motor_neurons("front", "right"))
    print(f"Front Right Motor Neurons: {len(motor_neurons):,}")
    right_side_vnc_neurons = uids.get_ids(uids.get_vnc_neurons("right"))
    print(f"Right Side VNC Neurons: {len(right_side_vnc_neurons):,}")
    upstream = descending_neurons + cpg_neurons
    downstream = motor_neurons
    allowed = right_side_vnc_neurons

    front_right_leg_subgraph = intermediate_subgraph(
        banc_graph,
        A=upstream,
        B=downstream,
        C=allowed,
        max_total_steps=2,
    )
    print(f"Nodes : {front_right_leg_subgraph.number_of_nodes():,}")
    print(f"Edges : {front_right_leg_subgraph.number_of_edges():,}")

    print(f"Saving graph to pickle")

    with open('generated/front_right_leg_subgraph.pkl', "wb") as f:
        pickle.dump(front_right_leg_subgraph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving graph to GraphML format")

    nx.write_graphml(front_right_leg_subgraph, 'generated/front_right_leg_subgraph.graphml')

    print("Done.")