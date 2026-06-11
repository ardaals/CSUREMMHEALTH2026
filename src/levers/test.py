from os.path import join
from connectome import Connectome
from parser import graphml_to_string, parse_graphml
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

def load_connectome_from_graphml(graphml_filename):
    graphml_dir = "./data/86node_connectomes"
    graphml_path = join(graphml_dir, graphml_filename)
    return Connectome(graphml_path)


connectome = load_connectome_from_graphml("100206_repeated10_scale33.graphml")
c1 = parse_graphml(graphml_to_string("./data/86node_connectomes/100206_repeated10_scale33.graphml"))
node_names = nx.get_node_attributes(c1, "dn_fsname")

