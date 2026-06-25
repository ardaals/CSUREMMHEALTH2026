import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import expm, solve, svd
from wc_criticality import default_wc_params, simulate_wc, wilson_cowan_rhs, sigmoid
import scipy.integrate as scipy_integrate
# Compatibility patch for nctpy with newer SciPy versions.
if not hasattr(scipy_integrate, "simps") and hasattr(scipy_integrate, "simpson"):
    scipy_integrate.simps = scipy_integrate.simpson

from nctpy.energies import get_control_inputs, integrate_u



def sigmoid_prime(x, a, theta):
    """
    Derivative of the shifted Wilson-Cowan sigmoid.

    The constant shift disappears when differentiating.
    """
    z = np.clip(-a * (x - theta), -500, 500)
    logistic = 1.0 / (1.0 + np.exp(z))
    return a * logistic * (1.0 - logistic)



def wc_jacobian(E_star, I_star, J, params, P=None):
    """
    Compute the deterministic Wilson-Cowan Jacobian.


    State ordering:
        x = [E_1, ..., E_N, I_1, ..., I_N]
    """

    N = J.shape[0]

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

    Smax_e = 1.0 - 1.0 / (1.0 + np.exp(aE * thetaE))
    Smax_i = 1.0 - 1.0 / (1.0 + np.exp(aI * thetaI))

    SEm = 2.0 * Smax_e
    SIm = 2.0 * Smax_i

    E_star = np.asarray(E_star, dtype=float)
    I_star = np.asarray(I_star, dtype=float)

    if P is None:
        P_t = np.zeros(N)
    else:
        P_t = np.asarray(P, dtype=float)

    eye = np.eye(N)

    hE = c1 * E_star - c2 * I_star + c5 * (J @ E_star) + P_t
    hI = c3 * E_star - c4 * I_star + c6 * (J @ I_star)

    S_E = sigmoid(hE, aE, thetaE)
    S_I = sigmoid(hI, aI, thetaI)

    S_E_prime = sigmoid_prime(hE, aE, thetaE)
    S_I_prime = sigmoid_prime(hI, aI, thetaI)

    D_E_gain = np.diag((SEm - r_e * E_star) * S_E_prime)
    D_I_gain = np.diag((SIm - r_i * I_star) * S_I_prime)

    A_EE = (
        -eye
        - r_e * np.diag(S_E)
        + D_E_gain @ (c1 * eye + c5 * J)
    ) / tau

    A_EI = (
        D_E_gain @ (-c2 * eye)
    ) / tau

    A_IE = (
        D_I_gain @ (c3 * eye)
    ) / tau

    A_II = (
        -eye
        - r_i * np.diag(S_I)
        + D_I_gain @ (-c4 * eye + c6 * J)
    ) / tau

    A = np.block([
        [A_EE, A_EI],
        [A_IE, A_II],
    ])

    return A



def get_near_critical_fixed_point(W, c5_operating, T=500.0, dt=0.5):
    """
    Simulate the Wilson-Cowan model at the pre-critical operating point.

    c5_operating should usually be:
        criticality_result["c5_pre"]

    The final simulated state is used as the near-critical fixed point.
    """

    params = default_wc_params()
    params["c5"] = float(c5_operating)
    params["c6"] = float(c5_operating / 4.0)

    t, E, I = simulate_wc(
        J=W,
        params=params,
        T=T,
        dt=dt,
    )

    E_star = E[:, -1]
    I_star = I[:, -1]

    y_star = np.concatenate([E_star, I_star])
    dydt = wilson_cowan_rhs(0.0, y_star, W, params)

    residual_norm = float(np.linalg.norm(dydt))

    post_start = E.shape[1] // 2
    fixed_point_post_std = float(
        E[:, post_start:].std() + I[:, post_start:].std()
    )

    return {
        "params": params,
        "E_star": E_star,
        "I_star": I_star,
        "y_star": y_star,
        "residual_norm": residual_norm,
        "fixed_point_post_std": fixed_point_post_std,
    }

def dominant_instability_mode(A):
    """
    Find the leading eigenvector of A.

    The target direction is the eigenvector whose eigenvalue has the largest real part.
    For a stable near-critical point, the largest real part should be negative but close to 0.
    """

    eigvals, eigvecs = np.linalg.eig(A)

    leading_index = int(np.argmax(np.real(eigvals)))

    leading_eigval = eigvals[leading_index]
    leading_vec = eigvecs[:, leading_index]

    # NCT target needs to be real-valued.
    target = np.real(leading_vec)

    # If the real part is accidentally tiny, use the imaginary part.
    if np.linalg.norm(target) < 1e-12:
        target = np.imag(leading_vec)

    target = np.asarray(target, dtype=float)

    norm = np.linalg.norm(target)
    if norm == 0:
        raise ValueError("Leading eigenvector has zero norm.")

    target = target / norm

    return target, leading_eigval


def make_single_region_B(n_state, control_coordinate):
    """
    Make an nctpy-compatible B matrix for one controlled coordinate.

    nctpy examples use B as an n_state x n_state matrix. For single-coordinate
    control, we make a diagonal matrix with exactly one nonzero control entry.
    """

    B = np.zeros((n_state, n_state), dtype=float)
    B[control_coordinate, control_coordinate] = 1.0
    return B


def nctpy_control_energy_for_region(
    A,
    target,
    control_coordinate,
    T_control=1.0,
    system="continuous",
    xr="zero",
    expm_version="eig",
    nctpy_dt=0.01,
    scale_energy_by_dt=True,
):
    """
    Compute minimum control energy for one controlled coordinate using nctpy.

    Important:
        This does NOT renormalize A.
        We pass the Wilson-Cowan Jacobian directly as A_norm=A.

    Parameters
    ----------
    A : ndarray
        Wilson-Cowan Jacobian.
    target : ndarray
        Target state, usually leading eigenvector of A.
    control_coordinate : int
        Coordinate receiving control input. For excitatory control of region i,
        this is i. For inhibitory control of region i, this is N + i.
    T_control : float
        Control horizon. T=1 with nctpy_dt=0.001 corresponds to ~1000 internal steps.
    system : str
        "continuous" for dx/dt = A x + B u.
    xr : str
        "zero" for minimum control setup.
    expm_version : str
        nctpy matrix exponential option.
    nctpy_dt : float
        nctpy examples treat dt as 0.001 when scaling energy.
    scale_energy_by_dt : bool
        If True, multiply summed energy by nctpy_dt. This follows the nctpy
        minimum-energy example that divides by 1000 for dt=0.001.
    """

    A = np.asarray(A, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1, 1)

    n_state = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if target.shape[0] != n_state:
        raise ValueError("target dimension must match A.")

    if control_coordinate < 0 or control_coordinate >= n_state:
        raise ValueError("control_coordinate is out of bounds.")

    x0 = np.zeros((n_state, 1), dtype=float)
    xf = target

    B = make_single_region_B(
        n_state=n_state,
        control_coordinate=control_coordinate,
    )

    # S = 0 means minimum control, not optimal control with trajectory constraint.
    S = np.zeros((n_state, n_state), dtype=float)

    x, u, n_err = get_control_inputs(
        A_norm=A,          # nctpy's parameter name; NOT renormalized here.
        T=T_control,
        B=B,
        x0=x0,
        xf=xf,
        system=system,
        xr=xr,
        S=S,
        expm_version=expm_version,
    )

    node_energy = np.asarray(integrate_u(u), dtype=float)
    raw_energy = float(np.sum(node_energy))

    if scale_energy_by_dt:
        energy = raw_energy * nctpy_dt
    else:
        energy = raw_energy

    n_err = np.asarray(n_err, dtype=float).ravel()

    inversion_error = float(n_err[0]) if n_err.size > 0 else np.nan
    reconstruction_error = float(n_err[1]) if n_err.size > 1 else np.nan

    active_coordinate_energy = float(node_energy[control_coordinate])
    if scale_energy_by_dt:
        active_coordinate_energy *= nctpy_dt

    return {
        "energy": float(energy),
        "raw_energy": raw_energy,
        "active_coordinate_energy": active_coordinate_energy,
        "inversion_error": inversion_error,
        "reconstruction_error": reconstruction_error,
        "u_shape": str(np.shape(u)),
        "x_shape": str(np.shape(x)),
    }



def prepare_node_metadata(nodes, node_metadata=None):
    """
    Prepare node metadata in the same order as W/nodes.

    node_metadata can be:
        None
        pandas DataFrame
        list of rows from node_attributes(graph)

    Expected list format from your node_attributes(graph):
        [node_name, node_id, degree, position, region, fsname, hemisphere, connected_nodes]
    """

    if node_metadata is None:
        df = pd.DataFrame({
            "region_index": np.arange(len(nodes)),
            "node": nodes,
        })
        return df

    if isinstance(node_metadata, pd.DataFrame):
        df = node_metadata.copy()
    else:
        df = pd.DataFrame(
            node_metadata,
            columns=[
                "node_name",
                "node_id",
                "degree_graph",
                "position",
                "region",
                "fsname",
                "hemisphere",
                "connected_nodes",
            ],
        )

    df = df.reset_index(drop=True)
    df.insert(0, "region_index", np.arange(len(df)))

    if "node" not in df.columns:
        df.insert(1, "node", nodes)

    if len(df) != len(nodes):
        raise ValueError(
            f"node_metadata has {len(df)} rows, but nodes has {len(nodes)} entries."
        )

    return df


def compute_connectivity_metrics(W):
    """
    Compute simple node-level connectivity metrics from W.

    These are useful for checking whether control energy is just tracking hubs.
    """

    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)

    binary_W = W > 0

    binary_degree = binary_W.sum(axis=1)
    weighted_degree = W.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_connection_weight = np.where(
            binary_degree > 0,
            weighted_degree / binary_degree,
            0.0,
        )

    max_connection_weight = W.max(axis=1)

    metrics_df = pd.DataFrame({
        "region_index": np.arange(W.shape[0]),
        "binary_degree": binary_degree.astype(int),
        "weighted_degree": weighted_degree,
        "mean_connection_weight": mean_connection_weight,
        "max_connection_weight": max_connection_weight,
    })

    metrics_df["weighted_degree_rank"] = metrics_df["weighted_degree"].rank(
        ascending=False,
        method="min",
    )

    metrics_df["binary_degree_rank"] = metrics_df["binary_degree"].rank(
        ascending=False,
        method="min",
    )

    return metrics_df


def make_state_metadata(nodes, node_metadata_df):
    """
    Create metadata for the 2N-dimensional Jacobian coordinates.

    If there are 83 regions, this returns 166 rows:
        E_0, ..., E_82, I_0, ..., I_82
    """

    rows = []
    N = len(nodes)

    for population in ["E", "I"]:
        for region_index, node_label in enumerate(nodes):
            meta = node_metadata_df.iloc[region_index].to_dict()

            coordinate = region_index if population == "E" else N + region_index

            row = {
                "state_coordinate": coordinate,
                "state_label": f"{population}_{region_index}_{node_label}",
                "population": population,
                "region_index": region_index,
                "node": node_label,
            }

            for key, value in meta.items():
                if key not in row:
                    row[key] = value

            rows.append(row)

    state_metadata_df = pd.DataFrame(rows)

    return state_metadata_df


def save_jacobian_excel(
    A,
    target,
    nodes,
    node_metadata_df,
    output_path,
):
    """
    Save the Jacobian and target mode to an Excel file with readable labels.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_metadata_df = make_state_metadata(nodes, node_metadata_df)

    state_labels = state_metadata_df["state_label"].tolist()

    A_df = pd.DataFrame(
        A,
        index=state_labels,
        columns=state_labels,
    )

    target_df = state_metadata_df.copy()
    target_df["target_mode_value"] = target

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        A_df.to_excel(writer, sheet_name="Jacobian_A")
        state_metadata_df.to_excel(writer, sheet_name="State_Metadata", index=False)
        target_df.to_excel(writer, sheet_name="Target_Mode", index=False)

    return str(output_path)


def save_network_control_excel(
    results_df,
    node_metadata_df,
    connectivity_metrics_df,
    output_path,
):
    """
    Save network-control results and node/connectivity metadata to Excel.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Control_Rankings", index=False)
        node_metadata_df.to_excel(writer, sheet_name="Node_Metadata", index=False)
        connectivity_metrics_df.to_excel(writer, sheet_name="Connectivity_Metrics", index=False)

    return str(output_path)


def run_network_control_analysis(
    W,
    nodes,
    criticality_result,
    output_directory,
    output_prefix,
    spectral_radius=False,
    T_control=1.0,
    fixed_point_T=None,
    fixed_point_dt=0.5,
    control="E",
    expm_version="eig",
    nctpy_dt=0.001,
    scale_energy_by_dt=True,
    save_arrays=True,
    node_metadata=None,
    save_excel=True,
    save_jacobian_excel_file=True,
):
    """
    Full Network Control Theory analysis for one connectome using nctpy.

    This version does NOT remove nodes and does NOT renormalize the Jacobian.

    Expected input:
        W is already the final connectome matrix, e.g. 83 x 83.

    Recommended criticality_result:
        output from run_criticality_analysis(...)

    The operating point is:
        criticality_result["c5_pre"]

    Parameters
    ----------
    W : ndarray
        Structural connectivity matrix. This should already be normalized before
        being passed here.
    nodes : list
        Node labels in the same order as W.
    criticality_result : dict
        Output dictionary from run_criticality_analysis.
    output_directory : str or Path
        Folder where outputs are saved.
    output_prefix : str
        Prefix for output files.
    spectral_radius : bool
        Used only for choosing faster defaults. This function does not normalize W.
    T_control : float
        NCT control horizon.
    fixed_point_T : float or None
        Simulation duration used to estimate the near-critical fixed point.
        If None, uses 300 for spectral_radius=True and 500 otherwise.
    fixed_point_dt : float
        Simulation time step used in the WCM simulation.
    control : {"E", "I"}
        Whether input is applied to the excitatory or inhibitory coordinate.
    expm_version : str
        nctpy matrix exponential option.
    nctpy_dt : float
        Energy scaling step. Default 0.001.
    scale_energy_by_dt : bool
        Whether to multiply summed nctpy energy by nctpy_dt.
    save_arrays : bool
        Whether to save A, target, E_star, and I_star as .npy files.

    Returns
    -------
    results_df : pandas.DataFrame
        Region-wise control energy rankings.
    """

    if "c5_pre" not in criticality_result:
        raise ValueError("criticality_result must contain 'c5_pre'.")

    c5_operating = float(criticality_result["c5_pre"])

    if fixed_point_T is None:
        if spectral_radius:
            fixed_point_T = 300.0
        else:
            fixed_point_T = 500.0

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)

    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square.")

    N = W.shape[0]

    if nodes is None:
        nodes = list(range(N))


    if len(nodes) != N:
        raise ValueError(
            f"nodes has length {len(nodes)}, but W has shape {W.shape}."
        )

    node_metadata_df = prepare_node_metadata(
        nodes=nodes,
        node_metadata=node_metadata,
    )

    connectivity_metrics_df = compute_connectivity_metrics(W)

    node_metadata_df = node_metadata_df.merge(
        connectivity_metrics_df,
        on="region_index",
        how="left",
    )

    if control not in ["E", "I"]:
        raise ValueError("control must be 'E' or 'I'.")
    # get the near-critical fixed point and Jacobian
    fixed = get_near_critical_fixed_point(
        W=W,
        c5_operating=c5_operating,
        T=fixed_point_T,
        dt=fixed_point_dt,
    )

    params = fixed["params"]
    E_star = fixed["E_star"]
    I_star = fixed["I_star"]
    # Compute the Jacobian at the near-critical operating point.
    A = wc_jacobian(
        E_star=E_star,
        I_star=I_star,
        J=W,
        params=params,
    )
    # Compute control energy for each region and save results.
    n_state = A.shape[0]

    if n_state != 2 * N:
        raise ValueError("Jacobian shape does not match 2N state size.")
    # Compute the leading eigenvalue for later interpretation.
    eigvals = np.linalg.eigvals(A)
    # This is the largest real part of any eigenvalue, which should be negative but close to 0 for a stable near-critical point.
    lambda_max_real = float(np.max(np.real(eigvals)))
    # This is the eigenvalue of the dominant instability mode, which should be the same as lambda_max_real but we save it separately for clarity.
    target, leading_eigval = dominant_instability_mode(A)
    # control_coordinate is the index of the state receiving input. For excitatory control of region i, this is i. For inhibitory control of region i, this is N + i.
    if control == "E":
        control_indices = np.arange(N)
    else:
        control_indices = N + np.arange(N)

    binary_degree = connectivity_metrics_df["binary_degree"].to_numpy()
    weighted_degree = connectivity_metrics_df["weighted_degree"].to_numpy()
    mean_connection_weight = connectivity_metrics_df["mean_connection_weight"].to_numpy()
    max_connection_weight = connectivity_metrics_df["max_connection_weight"].to_numpy()
    weighted_degree_rank = connectivity_metrics_df["weighted_degree_rank"].to_numpy()
    binary_degree_rank = connectivity_metrics_df["binary_degree_rank"].to_numpy()

    rows = []

    for region_index, node_label in enumerate(nodes):
        control_coordinate = int(control_indices[region_index])

        energy_result = nctpy_control_energy_for_region(
            A=A,
            target=target,
            control_coordinate=control_coordinate,
            T_control=T_control,
            system="continuous",
            xr="zero",
            expm_version=expm_version,
            nctpy_dt=nctpy_dt,
            scale_energy_by_dt=scale_energy_by_dt,
        )

        rows.append({
            "node_name": node_metadata_df["node_name"].iloc[region_index],
            "node": node_label,
            "control_coordinate": control_coordinate,
            "control_type": control,
            "energy": energy_result["energy"],
            "raw_energy": energy_result["raw_energy"],
            "active_coordinate_energy": energy_result["active_coordinate_energy"],
            "inversion_error": energy_result["inversion_error"],
            "reconstruction_error": energy_result["reconstruction_error"],
            "u_shape": energy_result["u_shape"],
            "x_shape": energy_result["x_shape"],
            "c5_operating": c5_operating,
            "c5_star": float(criticality_result["c5_star"]),
            "jump_size": float(criticality_result["jump_size"]),
            "T_control": float(T_control),
            "nctpy_dt": float(nctpy_dt),
            "estimated_control_steps": int(round(T_control / nctpy_dt)),
            "scale_energy_by_dt": bool(scale_energy_by_dt),
            "spectral_radius": bool(spectral_radius),
            "fixed_point_T": float(fixed_point_T),
            "fixed_point_dt": float(fixed_point_dt),
            "lambda_max_real": lambda_max_real,
            "leading_eigenvalue_real": float(np.real(leading_eigval)),
            "leading_eigenvalue_imag": float(np.imag(leading_eigval)),
            "fixed_point_residual_norm": fixed["residual_norm"],
            "fixed_point_post_std": fixed["fixed_point_post_std"],
            "binary_degree": int(binary_degree[region_index]),
            "weighted_degree": float(weighted_degree[region_index]),
            "mean_connection_weight": float(mean_connection_weight[region_index]),
            "max_connection_weight": float(max_connection_weight[region_index]),
            "weighted_degree_rank": float(weighted_degree_rank[region_index]),
            "binary_degree_rank": float(binary_degree_rank[region_index]),
        })

    results_df = pd.DataFrame(rows)

    results_df["rank"] = results_df["energy"].rank(method="min", ascending=True)
    results_df = results_df.sort_values("energy").reset_index(drop=True)

    csv_path = output_directory / f"{output_prefix}_network_control_rankings.csv"
    results_df.to_csv(csv_path, index=False)
    rankings_excel_path = None
    jacobian_excel_path = None

    if save_excel:
        rankings_excel_path = output_directory / f"{output_prefix}_network_control_rankings.xlsx"
        save_network_control_excel(
            results_df=results_df,
            node_metadata_df=node_metadata_df,
            connectivity_metrics_df=connectivity_metrics_df,
            output_path=rankings_excel_path,
        )

    if save_jacobian_excel_file:
        jacobian_excel_path = output_directory / f"{output_prefix}_jacobian_A.xlsx"
        save_jacobian_excel(
            A=A,
            target=target,
            nodes=nodes,
            node_metadata_df=node_metadata_df,
            output_path=jacobian_excel_path,
        )

    if save_arrays:
        np.save(output_directory / f"{output_prefix}_jacobian_A.npy", A)
        np.save(output_directory / f"{output_prefix}_target_mode.npy", target)
        np.save(output_directory / f"{output_prefix}_E_star.npy", E_star)
        np.save(output_directory / f"{output_prefix}_I_star.npy", I_star)

    return results_df