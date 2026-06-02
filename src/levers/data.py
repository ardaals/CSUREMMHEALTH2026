"""Part-1 pipeline: build frozen inputs from raw data sources.

Sections map to the analysis notebook:
  Section 4c  – canonical DK-68 region constants
  Section 4d  – load Budapest GraphML and crosswalk to DK-68
  Section 4e  – spectral normalization
  Section 5b  – FTD atrophy map (Zenodo NIfTI -> atrophy_FTD)
  Section 5c  – AD atrophy map  (ADNI tabular   -> atrophy_AD)
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker

# ---------------------------------------------------------------------------
# Paths (callers may pass explicit paths; these defaults let
# scripts/01_build_inputs.py work without arguments from the project root)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Section 4c – Canonical DK-68 region constants
# ---------------------------------------------------------------------------

DK34 = [
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
    'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal',
    'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual',
    'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral',
    'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
    'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
    'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal',
    'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole',
    'temporalpole', 'transversetemporal', 'insula',
]
assert len(DK34) == 34

# canonical 68-vector ordering: left hemisphere first, then right
CANON = [f'lh_{r}' for r in DK34] + [f'rh_{r}' for r in DK34]

# non-cortical Lausanne2008 scale33 labels to drop
# (note Lausanne spelling 'hyppocampus' alongside standard 'hippocampus')
NONCORTICAL = {
    'thalamusproper', 'caudate', 'putamen', 'pallidum', 'accumbensarea',
    'hyppocampus', 'hippocampus', 'amygdala', 'brainstem',
}


# ---------------------------------------------------------------------------
# Section 4d – Load Budapest GraphML and crosswalk to DK-68
# ---------------------------------------------------------------------------

def inspect_graphml(graphml_path):
    """Print node/edge attribute keys so you can confirm label names for your file."""
    G = nx.read_graphml(str(graphml_path))
    nid, ndata = next(iter(G.nodes(data=True)))
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("node attribute keys:", sorted(ndata.keys()))
    print("example node:", {k: ndata[k] for k in list(ndata)[:6]})
    if G.number_of_edges():
        u, v, edata = next(iter(G.edges(data=True)))
        print("edge attribute keys:", sorted(edata.keys()))
    return G


def _parse_node_attrs(G):
    """Return (label_key, hemi_key, parse_fn) for the node attributes in *G*."""
    _, first = next(iter(G.nodes(data=True)))
    attrs = set(first.keys())
    label_key = next((k for k in ['dn_name', 'dn_fsname', 'name', 'label'] if k in attrs), None)
    hemi_key  = next((k for k in ['dn_hemisphere', 'hemisphere', 'hemi'] if k in attrs), None)
    if label_key is None:
        raise KeyError(f"No label attribute found. Node attrs present: {sorted(attrs)}")

    def parse(d):
        raw  = str(d[label_key]).strip()
        hemi = str(d.get(hemi_key, '')).lower() if hemi_key else ''
        bare = raw
        for pre, h in [
            ('lh.', 'lh'), ('rh.', 'rh'),
            ('lh_', 'lh'), ('rh_', 'rh'),
            ('left-', 'lh'), ('right-', 'rh'),
            ('ctx-lh-', 'lh'), ('ctx-rh-', 'rh'),
        ]:
            if raw.lower().startswith(pre):
                bare = raw[len(pre):]
                hemi = h
                break
        if hemi in ('left', 'l'):  hemi = 'lh'
        if hemi in ('right', 'r'): hemi = 'rh'
        return hemi, bare.lower()

    return label_key, hemi_key, parse


def _dk68_index(G, parse):
    """Map each canonical DK-68 name to its row index in G's adjacency matrix.

    Returns (idx, nodelist) where idx[i] is the position of CANON[i] in nodelist.
    """
    nodelist = list(G.nodes())
    pos = {n: i for i, n in enumerate(nodelist)}

    name2nid = {}
    for nid, d in G.nodes(data=True):
        hemi, bare = parse(d)
        if bare in NONCORTICAL:
            continue
        if bare in DK34 and hemi in ('lh', 'rh'):
            name2nid[f'{hemi}_{bare}'] = nid

    missing = [c for c in CANON if c not in name2nid]
    if missing:
        raise ValueError(
            f"{len(missing)} DK regions unmatched (check labels), e.g. {missing[:5]}"
        )

    idx = [pos[name2nid[c]] for c in CANON]
    return idx, nodelist


def load_budapest_dk68(graphml_path, weight_attr='number_of_fibers'):
    """Return (W_68x68, names_68) on the canonical DK-68 cortical ordering.

    Parameters
    ----------
    graphml_path : path-like
        Either a single GraphML file or a directory of per-subject GraphML
        files (as in the Budapest connectome v3 HCP release). When a
        directory is given all *.graphml files are loaded and the fiber
        counts are averaged across subjects to produce a group connectome.
    weight_attr : str
        Edge attribute to use as connection weight.

    Returns
    -------
    W : (68, 68) float64 – (averaged) fiber-count matrix, symmetric, no diag
    names : list[str] – 68 canonical 'lh_<region>' / 'rh_<region>' labels
    """
    p = Path(graphml_path)

    if p.is_dir():
        files = sorted(p.glob('*.graphml'))
        if not files:
            raise FileNotFoundError(f"No .graphml files found in {p}")

        # Build the node mapping from the first subject (same topology for all)
        G0 = nx.read_graphml(str(files[0]))
        _, _, parse = _parse_node_attrs(G0)
        idx, nodelist = _dk68_index(G0, parse)

        # Accumulate fiber counts across subjects
        acc = np.zeros((68, 68), dtype=np.float64)
        for f in files:
            G = nx.read_graphml(str(f))
            A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight_attr)
            acc += A[np.ix_(idx, idx)]

        W = acc / len(files)
        print(f"Averaged {len(files)} subject connectomes.")

    else:
        G = nx.read_graphml(str(p))
        _, _, parse = _parse_node_attrs(G)
        idx, nodelist = _dk68_index(G, parse)
        A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight_attr)
        W = A[np.ix_(idx, idx)]

    W = np.maximum(W, W.T)   # enforce symmetry (undirected)
    np.fill_diagonal(W, 0.0) # remove self-connections
    return W, list(CANON)


# ---------------------------------------------------------------------------
# Section 4e – Spectral normalization
# ---------------------------------------------------------------------------

def normalize(W):
    """Scale W so its spectral radius equals 1.

    Under this normalization coupling strength g is comparable across
    connectomes and the linear critical point is g* = 1 / lambda_max = 1.
    """
    lambda_max = np.max(np.linalg.eigvals(W).real)
    return W / lambda_max


# ---------------------------------------------------------------------------
# Section 4b – Connectome pipeline (orchestrates 4d + 4e)
# ---------------------------------------------------------------------------

def build_connectome(
    graphml_path: Path | None = None,
    out_dir: Path | None = None,
    weight_attr: str = 'number_of_fibers',
) -> tuple[np.ndarray, list[str]]:
    """Load -> crosswalk -> normalize -> save.

    graphml_path may be a single GraphML file or the Budapest directory of
    per-subject files; when a directory is given subjects are averaged first.

    Writes
    ------
    data/processed/W_dk68.npy       – (68, 68) spectral-radius-1 connectome
    data/processed/region_names.npy – 68 canonical lh_/rh_ labels
    """
    graphml_path = Path(graphml_path) if graphml_path else RAW / "budapest_83.graphml"
    out_dir = Path(out_dir) if out_dir else PROCESSED

    W_raw, names = load_budapest_dk68(graphml_path, weight_attr=weight_attr)
    W = normalize(W_raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "W_dk68.npy", W)
    np.save(out_dir / "region_names.npy", np.array(names))
    return W, names


# ---------------------------------------------------------------------------
# Section 5b – FTD atrophy map
# ---------------------------------------------------------------------------

def parcellate_nifti_to_dk68(nifti_path: Path, atlas_img) -> np.ndarray:
    """Sample *nifti_path* values at DK-68 parcels and return a 68-vector.

    Parameters
    ----------
    nifti_path : Path  – bvFTD t-map NIfTI (Zenodo 10383493)
    atlas_img  : NIfTI-like – DK-68 parcellation in MNI space
    """
    # TODO: fill in atlas fetching (netneurotools / neuromaps) and masker logic
    raise NotImplementedError


def build_atrophy_ftd(
    nifti_path: Path | None = None,
    region_names: list[str] | None = None,
    out_dir: Path | None = None,
) -> np.ndarray:
    """Parcellate FTD t-map to DK-68 and save atrophy_FTD.npy.

    Writes
    ------
    data/processed/atrophy_FTD.npy
    """
    nifti_path = nifti_path or RAW / "ftd_bvFTD_tmap.nii.gz"
    out_dir = out_dir or PROCESSED

    # TODO: fetch DK-68 atlas via netneurotools / neuromaps
    atlas_img = None  # placeholder
    vec = parcellate_nifti_to_dk68(nifti_path, atlas_img)

    # sign convention: positive = more atrophy
    # TODO: confirm direction with source paper before finalizing
    atrophy = vec

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "atrophy_FTD.npy", atrophy)
    return atrophy


# ---------------------------------------------------------------------------
# Section 5c – AD atrophy map
# ---------------------------------------------------------------------------

def load_adni_tables(
    adni_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load UCSFFSX, DXSUM, and MMSE tables from *adni_dir*.

    Returns (thickness_df, diagnosis_df, mmse_df).
    """
    # TODO: confirm exact filenames once ADNI download is complete
    thickness = pd.read_csv(adni_dir / "UCSFFSX.csv", low_memory=False)
    diagnosis = pd.read_csv(adni_dir / "DXSUM_PDXCONV.csv", low_memory=False)
    mmse      = pd.read_csv(adni_dir / "MMSE.csv", low_memory=False)
    return thickness, diagnosis, mmse


def compute_atrophy_ad(
    thickness_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    mmse_df: pd.DataFrame,
    region_names: list[str],
) -> np.ndarray:
    """Derive a 68-vector of AD atrophy from ADNI cortical thickness.

    Strategy: (mean_CN - mean_AD) / pooled_std, so positive = more atrophy.
    """
    # TODO: implement group selection, ROI name mapping, and z-scoring
    raise NotImplementedError


def build_atrophy_ad(
    adni_dir: Path | None = None,
    region_names: list[str] | None = None,
    out_dir: Path | None = None,
) -> np.ndarray:
    """ADNI tables -> atrophy_AD.npy.

    Writes
    ------
    data/processed/atrophy_AD.npy
    """
    adni_dir = adni_dir or RAW / "adni"
    out_dir = out_dir or PROCESSED

    if region_names is None:
        region_names = list(np.load(out_dir / "region_names.npy", allow_pickle=True))

    thickness, diagnosis, mmse = load_adni_tables(adni_dir)
    atrophy = compute_atrophy_ad(thickness, diagnosis, mmse, region_names)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "atrophy_AD.npy", atrophy)
    return atrophy
