import numpy as np
import networkx as nx


def graphml_to_string(file):
    """Convert a GraphML file to a string."""
    with open(file, 'r') as f:
        return f.read()


def parse_graphml(graph_str):
    """Parse a GraphML string and return a NetworkX graph."""
    G = nx.parse_graphml(graph_str, node_type=int)
    return G
    
