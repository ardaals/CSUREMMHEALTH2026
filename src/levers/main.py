from parser import parse_graphml, graphml_to_string
from pathlib import Path
from util import get_files_in_directory, output_sheet
from connectome import edge_attributes, node_attributes
from os import path
from plot import plot_connectome_3d
from openpyxl import Workbook
from connectome import connectivity_matrix
import pandas as pd
from wc_criticality import run_criticality_analysis





def create_connectivity_matrix(source_directory, output_directory):
    files = get_files_in_directory(source_directory)
    for file in files:
        graph_str = graphml_to_string(path.join(source_directory,file))
        graph = parse_graphml(graph_str)
        W, nodes = connectivity_matrix(graph)
        df = pd.DataFrame(W, index=nodes, columns=nodes)
        # create output directory if it doesn't exist
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        df.to_csv(path.join(output_directory, file + ".csv"))



def node_info(source_directory, output_directory, output_file_name):
    # parameters: source_directory is the directory where the graphml files are located
    # output_directory is the directory where the output spreadsheet will be saved
    # output_file_name is the name of the output spreadsheet, without the .xlsx extension
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


def criticality_analysis(source_directory, output_directory):
    files = get_files_in_directory(source_directory)

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for file in files:
        print(f"{file}")
        graph_str = graphml_to_string(path.join(source_directory, file))
        graph = parse_graphml(graph_str)

        W, nodes = connectivity_matrix(graph)

        output_prefix = Path(file).stem

        result = run_criticality_analysis(
            W=W,
            output_directory=output_directory,
            output_prefix=output_prefix,
            c5_min=0.0,
            c5_max=1000.0,
            num_c5=101,
        )

        summary_rows.append({
            "file": file,
            "c5_star": result["c5_star"],
            "jump_size": result["jump_size"],
            "summary_csv_path": result["summary_csv_path"],
            "regional_csv_path": result["regional_csv_path"],
            "plot_path": result["plot_path"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        path.join(output_directory, "criticality_all_connectomes_summary.csv"),
        index=False
    )



if __name__ == '__main__':
    #put your code here

    pass
    # plot_graph("./data/86node_connectomes")
    # node_info("./data/dummy", "./spreadsheets", "node_info")
    create_connectivity_matrix("./data/dummy", "./data/dummy/connectivity_matrices")
    criticality_analysis(
        "./data/dummy",
        "./data/dummy/criticality1"
    )
