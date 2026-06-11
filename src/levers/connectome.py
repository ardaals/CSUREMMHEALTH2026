from parser import parse_graphml, graphml_to_string
import networkx as nx
from util import float_to_rounded_string


def node_attributes(graph):
    """Extract node attributes from a NetworkX graph."""
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

def edge_attributes(graph):
    """Extract edge attributes from a NetworkX graph."""
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

