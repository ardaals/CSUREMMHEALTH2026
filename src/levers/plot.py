import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def style_connectome_plot(fig, ax, nx_graph, nodes, degrees, betweenness_values, title_text=None):
    '''
    Apply professional styling and enhancements to the 3D connectome plot.
    Handles colorbar, legend, labels, and interactive zoom functionality.
    '''
    scatter = ax.collections[0]
    cbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.5)
    cbar.set_label('Betweenness', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel('X Position', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z Position', fontsize=10, fontweight='bold')
    if title_text:
        ax.set_title(f'3D Connectome Network — {title_text}', fontsize=12, fontweight='bold', pad=15)
    else:
        ax.set_title('3D Connectome Network', fontsize=12, fontweight='bold', pad=15)
    ax.view_init(elev=20, azim=45)

    min_degree = int(np.min(degrees))
    max_degree = int(np.max(degrees))
    mean_degree = int(np.mean(degrees))
    median_degree = int(np.median(degrees))

    legend_text = (
        '━━━ NODE METRICS ━━━\n'
        f'Size: Node Degree\n'
        f'  Min: {min_degree}  |  Max: {max_degree}\n'
        f'  Mean: {mean_degree}  |  Median: {median_degree}\n\n'
        '━━━ EDGE METRICS ━━━\n'
        f'Gray lines: Connections\n'
        f'Total edges: {nx_graph.number_of_edges()}\n\n'
        '━━━ NETWORK ━━━\n'
        f'Nodes: {len(nodes)}\n'
        f'Color: Betweenness\n'
        f'  Centrality'
    )

    fig.text(0.98, 0.97, legend_text, transform=fig.transFigure,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0',
                      edgecolor='#333333', linewidth=1.5, alpha=0.95))

    def on_scroll(event):
        if event.inaxes != ax:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_zlim = ax.get_zlim()
        xdata, ydata = event.xdata, event.ydata

        if event.button == 'up':
            scale_factor = 0.8
        elif event.button == 'down':
            scale_factor = 1.2
        else:
            return

        new_xlim = [xdata - (xdata - cur_xlim[0]) * scale_factor,
                    xdata + (cur_xlim[1] - xdata) * scale_factor]
        new_ylim = [ydata - (ydata - cur_ylim[0]) * scale_factor,
                    ydata + (cur_ylim[1] - ydata) * scale_factor]
        z_center = (cur_zlim[0] + cur_zlim[1]) / 2
        z_range = (cur_zlim[1] - cur_zlim[0]) / 2
        new_zlim = [z_center - z_range * scale_factor,
                    z_center + z_range * scale_factor]

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.set_zlim(new_zlim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)


def plot_connectome_3d(nx_graph, file_id=None):
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

    # Calculate node degrees for sizing
    degrees = np.array([nx_graph.degree(node) for node in nodes])
    
    # Calculate betweenness centrality for coloring (shows information flow importance)
    betweenness = nx.betweenness_centrality(nx_graph)
    betweenness_values = np.array([betweenness[node] for node in nodes])
    
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, projection="3d")
    
    # Set background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    # Plot edges with better visibility
    for edge in edge_xyz:
        ax.plot(*edge.T, color='gray', alpha=0.5, linewidth=1.0)

    # Plot nodes with size based on degree and color based on betweenness centrality
    ax.scatter(
        *node_xyz.T,
        c=betweenness_values,
        s=degrees * 3 + 15,  # Scaled down node sizes (based on degree)
        cmap='plasma',
        linewidth=0.5,
        alpha=0.8,
        edgecolors='white'
    )

    # Add node labels with better visibility
    for node, (x, y, z) in pos.items():
        ax.text(x, y, z, str(node), fontsize=6, alpha=0.5, style='italic')

    # Allow quitting from the figure with 'q' and capture state
    state = {"quit": False}

    def on_key(event):
        # If user presses 'q' inside the figure, close and mark quit
        if event.key == 'q':
            state['quit'] = True
            plt.close(event.canvas.figure)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Apply all styling and visualization enhancements (pass file id for title)
    style_connectome_plot(fig, ax, nx_graph, nodes, degrees, betweenness_values, title_text=file_id)

    fig.tight_layout()
    plt.show()

    return state['quit']
