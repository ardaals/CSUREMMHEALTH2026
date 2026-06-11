import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_connectome_3d(nx_graph):
# plots connectome in 3D using node positions from graph attributes
    pos = {
    node: (
        float(data["dn_position_x"]),
        float(data["dn_position_y"]),
        float(data["dn_position_z"])
    )
    for node, data in nx_graph.nodes(data=True)
    }

    nodes = list(nx_graph.nodes())

    node_xyz = np.array([pos[v] for v in nodes])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in nx_graph.edges()])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*node_xyz.T, s=100, ec="w")

    for node, (x, y, z) in pos.items():
        ax.text(x, y, z, str(node), fontsize=8)

    for edge in edge_xyz:
        ax.plot(*edge.T, color="tab:gray")



    fig.tight_layout()
    plt.show()