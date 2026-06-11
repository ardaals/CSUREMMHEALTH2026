from parser import parse_graphml, graphml_to_string
import networkx as nx


class Connectome:
    def __init__(self, file_path = None):
        self.TheConnectome = None
        self.ThePath = None
        if file_path != None:
            graph_str = graphml_to_string(file_path)
            self.TheConnectome = parse_graphml(graph_str)


