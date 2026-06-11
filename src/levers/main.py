from parser import parse_graphml, graphml_to_string
from util import get_files_in_directory
from connectome import Connectome
from os import path
from plot import plot_connectome_3d




def init_connectome(source_directory):

    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
        connectome = Connectome(graph)


def plot_graph(source_directory):
    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
        plot_connectome_3d(graph)


if __name__ == '__main__':
    #put your code here

    pass
    plot_graph("./data/86node_connectomes")