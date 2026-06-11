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
        node_info.append([
            node_name,
            node_id,
            degree,
            position,
            region,
            fsname,
            hemisphere
        ])
    return node_info

