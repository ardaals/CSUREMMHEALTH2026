from parser import parse_graphml, graphml_to_string
import networkx as nx
from util import float_to_rounded_string
import numpy as np


def node_attributes(graph):
    """Extract node attributes from a NetworkX graph."""
    # node_info[0] = node_name, node_info[1] = node_id, node_info[2] = degree, 
    # node_info[3] = position, node_info[4] = region, node_info[5] = fsname, 
    # node_info[6] = hemisphere, node_info[7] = connected_nodes
    node_info = []
    for node, data in graph.nodes(data=True):
        node_name = data.get("dn_name")
        node_id = node
        degree = graph.degree(node)
        position = f"{float_to_rounded_string(data.get('dn_position_x'))}, {float_to_rounded_string(data.get('dn_position_y'))}, {float_to_rounded_string(data.get('dn_position_z'))}"
        region = data.get("dn_region")
        fsname = data.get("dn_fsname")
        hemisphere = data.get("dn_hemisphere")
        connected_nodes = ", ".join(str(neighbor) for neighbor in graph.neighbors(node))
        node_info.append([
            node_name,
            node_id,
            degree,
            position,
            region,
            fsname,
            hemisphere,
            connected_nodes

        ])
    return node_info

def edge_validation(graph):
    node_info = []
    for node, data in graph.nodes(data=True):
        node_name = data.get("dn_name")
        if graph.degree(node) == 0:
            node_info.append(node_name)
    full_node_info = ", ".join(node_info)
    return full_node_info

def edge_attributes(graph):
    """Extract edge attributes from a NetworkX graph."""
    #edge_info[0] = source_node, edge_info[1] = target_node, edge_info[2] = fiber_length_mean,
    #edge_info[3] = fa_mean, edge_info[4] = number_of_fibers
    edge_info = []
    for u, v, data in graph.edges(data=True):
        fiber_length_mean = data.get("fiber_length_mean")
        fa_mean = data.get("FA_mean")
        num_fibers = data.get("number_of_fibers")
        edge_info.append([
            u,
            v,
            fiber_length_mean,
            fa_mean,
            num_fibers
        ])
    return edge_info


def connectivity_matrix(graph, spectral_radius=False, weight_attr="number_of_fibers", zero_diagonal=True):
    """Extract the connectivity matrix from a NetworkX graph."""
    nodes = list(graph.nodes())

    # Convert edge weights to floats
    for u, v, data in graph.edges(data=True):
        if weight_attr not in data:
            raise ValueError(f"Edge ({u}, {v}) missing attribute '{weight_attr}'")
        data[weight_attr] = float(data[weight_attr])

    # Build raw weighted adjacency matrix
    W = nx.to_numpy_array(graph, nodelist=nodes, weight=weight_attr, dtype=float)

    # Structural connectomes are usually treated as undirected so we symmetrize the matrix
    W  = (W + W.T) / 2

    # Remove self-connections
    if zero_diagonal:
        np.fill_diagonal(W, 0)

    total = W.sum()
    if total == 0:
            raise ValueError("Matrix sum is zero; check edge weights.")
    W = W / total
    if spectral_radius == True:
        W = W / np.max(np.abs(np.linalg.eigvals(W)))
    return W, nodes
