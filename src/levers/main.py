from parser import parse_graphml, graphml_to_string
from pathlib import Path
from util import get_files_in_directory, output_sheet
from connectome import node_attributes
from os import path
from plot import plot_connectome_3d
from openpyxl import Workbook





def init_connectome(source_directory):

    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
      


def node_info(source_directory, output_directory, output_file_name):
    wb = Workbook()
    ws = wb.active
    ws.append(["Node Table", "Node ID", "Node Degree", "Position", "Region", "FreeSurfer Name", "Hemisphere"])
    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
        rows = node_attributes(graph)
        for output in rows:
            ws.append(output)
    output_sheet(output_directory, output_file_name, wb)
        


def plot_graph(source_directory):
    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
        plot_connectome_3d(graph)


if __name__ == '__main__':
    #put your code here

    pass
    # plot_graph("./data/dummy")
    node_info("./data/dummy", "./spreadsheets", "node_info")