import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path


def sigmoid(x, a, theta):
    """
    Wilson-Cowan 1972 shifted sigmoid.

    S(x) = 1 / (1 + exp(-a(x - theta))) - 1 / (1 + exp(a theta))
    """
    z = np.clip(-a * (x - theta), -500, 500)
    return 1.0 / (1.0 + np.exp(z)) - 1.0 / (1.0 + np.exp(a * theta))


def wilson_cowan_rhs(t, y, J, params, P=None):
    """
    Noise-free, zero-delay Wilson-Cowan model on a connectome.

    State vector:
        y = [E_1, ..., E_N, I_1, ..., I_N]

    Model:
        tau dE_i/dt =
            -E_i + (2*Smax_E - r_e*E_i)
            * S_E(c1*E_i - c2*I_i + c5*sum_j J_ij E_j + P_i)

        tau dI_i/dt =
            -I_i + (2*Smax_I - r_i*I_i)
            * S_I(c3*E_i - c4*I_i + c6*sum_j J_ij I_j + Q_i)
    """

    N = J.shape[0]

    E = y[:N]
    I = y[N:]

    tau = params["tau"]

    r_e = params["r_e"]
    r_i = params["r_i"]

    c1 = params["c1"]
    c2 = params["c2"]
    c3 = params["c3"]
    c4 = params["c4"]
    c5 = params["c5"]
    c6 = params["c6"]

    aE = params["aE"]
    thetaE = params["thetaE"]

    aI = params["aI"]
    thetaI = params["thetaI"]

    # Maximum of the shifted sigmoid
    Smax_e = 1.0 - 1.0 / (1.0 + np.exp(aE * thetaE))
    Smax_i = 1.0 - 1.0 / (1.0 + np.exp(aI * thetaI))

    # MATLAB implementation uses 2*Smax_e and 2*Smax_i
    SEm = 2.0 * Smax_e
    SIm = 2.0 * Smax_i

    if P is None:
        P_t = np.zeros(N)
    elif callable(P):
        P_t = np.asarray(P(t))
    else:
        P_t = np.asarray(P)



    input_E = c1 * E - c2 * I + c5 * (J @ E) + P_t
    input_I = c3 * E - c4 * I + c6 * (J @ I) 

    S_E = sigmoid(input_E, a=aE, theta=thetaE)
    S_I = sigmoid(input_I, a=aI, theta=thetaI)

    dE = (-E + (SEm - r_e * E) * S_E) / tau
    dI = (-I + (SIm - r_i * I) * S_I) / tau

    return np.concatenate([dE, dI])


def simulate_wc(J, params, T=300.0, dt=0.5, E0_value=0.1, I0_value=0.1, P=None):
    """
    Simulate Wilson-Cowan dynamics for one value of c5.
    """

    N = J.shape[0]
    # Initial conditions: small random values around E0_value and I0_value
    y0 = np.concatenate([
        E0_value * np.ones(N),
        I0_value * np.ones(N)
    ])
    # Time points to evaluate the solution at
    t_eval = np.arange(0, T + dt, dt)
    # Integrate the system of ODEs
    sol = solve_ivp(
        fun=wilson_cowan_rhs,
        t_span=(0, T),
        y0=y0,
        t_eval=t_eval,
        args=(J, params, P),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(sol.message)
    # Extract E and I from the solution
    E = sol.y[:N, :]
    I = sol.y[N:, :]

    return sol.t, E, I


def summarize_activity(E, transient_fraction=0.5):
    """
    Summarize post-transient excitatory activity without using arbitrary thresholds.

    Returns the average excitatory activity for each region after stabilization.
    """
    # Assume the first transient_fraction of the time series is transient, and analyze the rest.
    start = int(transient_fraction * E.shape[1])
    E_post = E[:, start:]
    # Compute mean and std of E for each node across the post-transient time points
    E_mean_by_node = E_post.mean(axis=1)
    E_std_by_node = E_post.std(axis=1)

    return {
        "mean_activity": float(E_mean_by_node.mean()),
        "max_activity": float(E_mean_by_node.max()),
        "mean_oscillation": float(E_std_by_node.mean()),
        "E_mean_by_node": E_mean_by_node,
        "E_std_by_node": E_std_by_node,
    }


def sweep_c5(
    J,
    c5_values,
    base_params=None,
    c6_ratio=0.25,
    T=300.0,
    dt=0.5,
    transient_fraction=0.5,
):
    """
    Sweep c5 and record regional average excitatory activity.

    This avoids arbitrary high/oscillation thresholds.
    """

    if base_params is None:
        base_params = default_wc_params()
    
    summary_rows = []
    regional_rows = []
    
    for c5 in c5_values:
        params = base_params.copy()
        params["c5"] = float(c5)
        params["c6"] = float(c6_ratio * c5)

        t, E, I = simulate_wc(J, params, T=T, dt=dt)

        summary = summarize_activity(
            E,
            transient_fraction=transient_fraction,
        )

        summary_rows.append({
            "c5": float(c5),
            "c6": float(params["c6"]),
            "mean_activity": summary["mean_activity"],
            "max_activity": summary["max_activity"],
            "mean_oscillation": summary["mean_oscillation"],
        })

        regional_row = {
            "c5": float(c5),
            "c6": float(params["c6"]),
        }

        for i, value in enumerate(summary["E_mean_by_node"]):
            regional_row[f"region_{i}"] = float(value)

        regional_rows.append(regional_row)

    summary_df = pd.DataFrame(summary_rows)
    regional_df = pd.DataFrame(regional_rows)

    return summary_df, regional_df


def estimate_critical_c5(regional_df):
    """
    Estimate c5* as the c5 value where the regional activity pattern
    has the largest jump.

    This is threshold-free.
    """

    c5_values = regional_df["c5"].to_numpy()
    region_cols = [col for col in regional_df.columns if col.startswith("region_")]

    R = regional_df[region_cols].to_numpy()

    # Difference in regional average activity from one c5 to the next
    regional_jumps = np.diff(R, axis=0)

    # One scalar jump score per c5 interval
    jump_scores = np.linalg.norm(regional_jumps, axis=1)
    # The jump size is the norm of the change in regional activity patterns from one c5 to the next.
    jump_index = int(np.argmax(jump_scores))
    # The estimated critical c5* is the c5 value at the end of the interval with the largest jump.
    c5_star = c5_values[jump_index + 1]
    jump_size = jump_scores[jump_index]

    return float(c5_star), float(jump_size)



def default_wc_params():
    """
    Default starting parameters based on Kora et. al..

    """

    return {
        "tau": 8.0,

        "r_e": 1.0,
        "r_i": 1.0,


        "c1": 16.0,
        "c2": 12.0,
        "c3": 15.0,
        "c4": 3.0,

        "c5": 0.0,
        "c6": 0.0,

        "aE": 1.3,
        "thetaE": 4.0,

        "aI": 2.0,
        "thetaI": 3.7,
    }


def plot_criticality(summary_df, regional_df, c5_star, output_path):
    """
    Plot regional average activity curves and the estimated transition point.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c5_values = regional_df["c5"].to_numpy()
    region_cols = [col for col in regional_df.columns if col.startswith("region_")]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Plot each region's average excitatory activity as a function of c5
    for col in region_cols:
        ax1.plot(
            c5_values,
            regional_df[col],
            linewidth=0.7,
            alpha=0.35,
        )

    # Also plot the across-region mean as a thicker line
    ax1.plot(
        summary_df["c5"],
        summary_df["mean_activity"],
        linewidth=2.0,
        label="Mean across regions",
    )
    # Mark the estimated c5* with a vertical line
    ax1.axvline(
        c5_star,
        linestyle=":",
        linewidth=2,
        label=f"Estimated c5* = {c5_star:.4g}",
    )

    ax1.set_xlabel("Global excitatory coupling c5")
    ax1.set_ylabel("Regional average excitatory activity")
    ax1.set_title("Wilson-Cowan transition sweep")
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_criticality_analysis(
    W,
    output_directory,
    output_prefix,
    c5_min=0.0,
    c5_max=1000.0,
    num_c5=101,
):
    """
    Full criticality workflow for one connectome matrix.

    Outputs:
        summary CSV
        regional activity CSV
        transition plot
        estimated c5*
    """

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)

    c5_values = np.linspace(c5_min, c5_max, num_c5)

    summary_df, regional_df = sweep_c5(
        J=W,
        c5_values=c5_values,
        base_params=default_wc_params(),
        c6_ratio=0.25,
        T=300.0,
        dt=0.5,
        transient_fraction=0.5,
    )

    c5_star, jump_size = estimate_critical_c5(regional_df)

    summary_csv_path = Path(output_directory) / f"{output_prefix}_criticality_summary.csv"
    regional_csv_path = Path(output_directory) / f"{output_prefix}_regional_activity.csv"
    plot_path = Path(output_directory) / f"{output_prefix}_criticality_plot.png"

    summary_df.to_csv(summary_csv_path, index=False)
    regional_df.to_csv(regional_csv_path, index=False)

    plot_criticality(summary_df, regional_df, c5_star, plot_path)
    timeseries_paths = plot_timeseries_near_transition(
        W=W,
        c5_star=c5_star,
        output_directory=output_directory,
        output_prefix=output_prefix,
        below_delta=(c5_max - c5_min) / max(num_c5 - 1, 1),
        above_delta=(c5_max - c5_min) / max(num_c5 - 1, 1),
        T=300.0,
        dt=0.5,
    )

    return {
        "c5_star": c5_star,
        "jump_size": jump_size,
        "csv_path": str(summary_csv_path),
        "summary_csv_path": str(summary_csv_path),
        "regional_csv_path": str(regional_csv_path),
        "plot_path": str(plot_path),
        "timeseries_paths": timeseries_paths,
    }


def plot_timeseries_near_transition(
    W,
    c5_star,
    output_directory,
    output_prefix,
    below_delta=10.0,
    above_delta=10.0,
    T=300.0,
    dt=0.5,
):
    """
    Plot E(t) time series below, at, and above the estimated transition.

    This is useful for verifying that c5* corresponds to a true dynamical
    transition rather than just a plotting artifact.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    c5_values = [
        max(c5_star - below_delta, 0.0),
        c5_star,
        c5_star + above_delta,
    ]

    labels = [
        f"below_c5star_{c5_values[0]:.4g}",
        f"at_c5star_{c5_values[1]:.4g}",
        f"above_c5star_{c5_values[2]:.4g}",
    ]

    output_paths = []

    for c5, label in zip(c5_values, labels):
        params = default_wc_params()
        params["c5"] = float(c5)
        params["c6"] = float(c5 / 4.0)

        t, E, I = simulate_wc(
            W,
            params,
            T=T,
            dt=dt,
        )

        plot_path = output_directory / f"{output_prefix}_timeseries_{label}.png"

        plt.figure(figsize=(9, 5))

        for node_idx in range(E.shape[0]):
            plt.plot(
                t,
                E[node_idx, :],
                linewidth=0.7,
                alpha=0.45,
            )

        plt.xlabel("Time")
        plt.ylabel("Excitatory activity E(t)")
        plt.title(f"Wilson-Cowan E(t), c5 = {c5:.4g}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        output_paths.append(str(plot_path))

    return output_paths