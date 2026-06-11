from parser import parse_graphml, graphml_to_string
from pathlib import Path
from util import get_files_in_directory, output_sheet
from connectome import edge_attributes, node_attributes
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
    ws1 = wb.active
    ws1.append(["Node Table", "Node ID", "Node Degree", "Position", "Region", "FreeSurfer Name", "Hemisphere", "Connected Nodes"])
    ws2 = wb.create_sheet("Edge Table")
    ws2.append(["Source Node ID", "Target Node ID", "Fiber Length Mean", "Fractional Anisotropy Mean", "Number of Fibers"])
    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
        rows_node = node_attributes(graph)
        edge_info = edge_attributes(graph)
        for output in rows_node:
            ws1.append(output)
        for output in edge_info:
            ws2.append(output)
    output_sheet(output_directory, output_file_name, wb)
        


def plot_graph(source_directory):
    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
        # extract numeric id from filename (before first underscore)
        brain_id = Path(file).stem.split('_')[0]

        # show plot and allow quitting with 'q' inside the figure window
        quit_flag = plot_connectome_3d(graph, file_id=brain_id)
        if quit_flag:
            break

        # Prompt user in console after the figure closes
        print(f"Displayed {file}. Press 'q' to quit, or press Enter to continue: ", end='', flush=True)
        response = input()
        if response.lower() == 'q':
            break


if __name__ == '__main__':
    #put your code here

    pass
    plot_graph("./data/86node_connectomes")
    # node_info("./data/dummy", "./spreadsheets", "node_info")