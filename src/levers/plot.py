import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

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




def plot_lever_connectivity_spearman(
    results_df,
    connectivity_metric,
    output_directory,
    output_prefix,
    lever_rank_column="rank",
    use_connectivity_rank=True,
    annotate_top_n=0,
):
    """
    Plot Spearman rank correlation between lever rank and a connectivity metric.

    results_df should be the output from run_network_control_analysis.

    If use_connectivity_rank=True:
        x-axis = connectivity rank, where 1 = highest connectivity
        y-axis = lever rank, where 1 = lowest control energy

    In that setup:
        positive rho means high-connectivity nodes tend to be strong levers.
        rho near 0 means lever rank is not explained by connectivity rank.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    df = results_df.copy()

    if lever_rank_column not in df.columns:
        raise ValueError(f"Missing lever rank column: {lever_rank_column}")

    if connectivity_metric not in df.columns:
        raise ValueError(f"Missing connectivity metric column: {connectivity_metric}")

    if use_connectivity_rank:
        connectivity_rank_column = f"{connectivity_metric}_spearman_rank"

        df[connectivity_rank_column] = df[connectivity_metric].rank(
            ascending=False,
            method="min",
        )

        x_column = connectivity_rank_column
        x_label = f"{connectivity_metric} rank (1 = highest connectivity)"
    else:
        x_column = connectivity_metric
        x_label = connectivity_metric

    y_column = lever_rank_column
    y_label = "Lever rank (1 = lowest control energy)"

    keep_columns = ["node", x_column, y_column]
    # Add region name column if it exists
    if "region_name" in df.columns:
        keep_columns.append("region_name")
    elif "region" in df.columns:
        keep_columns.append("region")
    elif "fsname" in df.columns:
        keep_columns.append("fsname")
    # plot_df will be used for plotting and correlation calculation, so we drop rows with NaN or Inf values in the relevant columns
    plot_df = (
        df[keep_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[x_column, y_column])
    )

    if len(plot_df) < 3:
        raise ValueError(
            f"Not enough valid rows to compute Spearman correlation for {connectivity_metric}."
        )

    rho, p_value = spearmanr(
        plot_df[x_column].to_numpy(),
        plot_df[y_column].to_numpy(),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    # Scatter plot of lever rank vs connectivity metric
    ax.scatter(
        plot_df[x_column],
        plot_df[y_column],
        alpha=0.8,
    )

    # Visual guide only. Spearman is rank-based; this line is not the test itself.
    slope, intercept = np.polyfit(
        plot_df[x_column].to_numpy(),
        plot_df[y_column].to_numpy(),
        1,
    )

    x_line = np.linspace(
        plot_df[x_column].min(),
        plot_df[x_column].max(),
        100,
    )

    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(
        f"Lever rank vs {connectivity_metric}\n"
        f"Spearman rho = {rho:.3f}, p = {p_value:.3g}, n = {len(plot_df)}"
    )

    ax.grid(True, alpha=0.3)
    
    if annotate_top_n > 0:
        label_df = plot_df.sort_values(y_column).head(annotate_top_n)

        for _, row in label_df.iterrows():
            if "region_name" in label_df.columns and pd.notna(row["region_name"]):
                label = str(row["region_name"])
            elif "region" in label_df.columns and pd.notna(row["region"]):
                label = str(row["region"])
            elif "fsname" in label_df.columns and pd.notna(row["fsname"]):
                label = str(row["fsname"])
            else:
                label = str(row["node"])

            ax.annotate(
                label,
                (row[x_column], row[y_column]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    fig.tight_layout()

    plot_path = (
        output_directory
        / f"{output_prefix}_lever_rank_vs_{connectivity_metric}_spearman.png"
    )

    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    return {
        "connectivity_metric": connectivity_metric,
        "x_column": x_column,
        "y_column": y_column,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "n_regions": int(len(plot_df)),
        "plot_path": str(plot_path),
        "use_connectivity_rank": bool(use_connectivity_rank),
    }


def plot_all_lever_connectivity_spearman(
    results_df,
    output_directory,
    output_prefix,
    connectivity_metrics=None,
    lever_rank_column="rank",
    use_connectivity_rank=True,
    annotate_top_n=0,
):
    """
    Run Spearman correlations between lever rank and multiple connectivity metrics.

    Saves:
        one PNG plot per metric
        one summary CSV
    """

    if connectivity_metrics is None:
        connectivity_metrics = [
            "weighted_degree",
            "binary_degree",
            "mean_connection_weight",
            "max_connection_weight",
        ]

    rows = []

    for metric in connectivity_metrics:
        if metric not in results_df.columns:
            print(f"Skipping {metric}: column not found.")
            continue

        result = plot_lever_connectivity_spearman(
            results_df=results_df,
            connectivity_metric=metric,
            output_directory=output_directory,
            output_prefix=output_prefix,
            lever_rank_column=lever_rank_column,
            use_connectivity_rank=use_connectivity_rank,
            annotate_top_n=annotate_top_n,
        )

        rows.append(result)

    correlation_df = pd.DataFrame(rows)

    summary_path = (
        Path(output_directory)
        / f"{output_prefix}_lever_connectivity_spearman_summary.csv"
    )

    correlation_df.to_csv(summary_path, index=False)

    return correlation_df