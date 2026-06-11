import re
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# Canonical DK34 and CANON keys (lh_/rh_)
DK34 = [
    "bankssts","caudalanteriorcingulate","caudalmiddlefrontal","cuneus","entorhinal",
    "frontalpole","fusiform","inferiorparietal","inferiortemporal","isthmuscingulate",
    "lateraloccipital","lateralorbitofrontal","lingual","medialorbitofrontal","middletemporal",
    "paracentral","parahippocampal","parsopercularis","parsorbitalis","parstriangularis",
    "pericalcarine","postcentral","posteriorcingulate","precentral","precuneus",
    "rostralanteriorcingulate","rostralmiddlefrontal","superiorfrontal","superiorparietal",
    "superiortemporal","supramarginal","temporalpole","transversetemporal","insula",
]

CANON = [f"lh_{r}" for r in DK34] + [f"rh_{r}" for r in DK34]

# FreeSurfer ST mapping provided (from plan)
ST_L = dict(zip(DK34, [13,14,15,23,24,25,26,31,32,34,35,36,38,39,40,
                       43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,62,129]))
ST_R = dict(zip(DK34, [72,73,74,82,83,84,85,90,91,93,94,95,97,98,99,
                       102,103,104,105,106,107,108,109,110,111,113,114,115,116,117,118,119,121,130]))

FS_TA = {**{f"lh_{r}": f"ST{ST_L[r]}TA" for r in DK34},
         **{f"rh_{r}": f"ST{ST_R[r]}TA" for r in DK34}}


def load_csvs(tau_path: str, fsx_path: str, amy_path: str, mrg_path: str):
    tau = pd.read_csv(tau_path, low_memory=False)
    fsx = pd.read_csv(fsx_path, low_memory=False)
    amy = pd.read_csv(amy_path, low_memory=False)
    mrg = pd.read_csv(mrg_path, low_memory=False)
    return tau, fsx, amy, mrg


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    if "VISCODE2" in df.columns:
        df["VISIT"] = df["VISCODE2"]
    elif "VISCODE" in df.columns:
        df["VISIT"] = df["VISCODE"]
    for d in ("SCANDATE", "EXAMDATE"):
        if d in df.columns:
            df[d] = pd.to_datetime(df[d], errors="coerce")
    return df


def dedup(df: pd.DataFrame, datecol: str) -> pd.DataFrame:
    keys = [k for k in ("RID", "VISIT") if k in df.columns]
    if datecol not in df.columns:
        return df.drop_duplicates(subset=keys, keep="last")
    return df.sort_values(datecol).drop_duplicates(subset=keys, keep="last")


def tau_col_for_key(tau_df: pd.DataFrame, canon_key: str) -> Optional[str]:
    hemi, region = canon_key.split("_", 1)
    # common patterns: CTX_LH_ENTORHINAL_SUVR or LH_ENTORHINAL_SUVR
    patterns = [rf"CTX_{hemi.upper()}_{region.upper()}_SUVR$",
                rf"{hemi.upper()}_{region.upper()}_SUVR$",
                rf"{hemi.upper()}_{region.upper()}_MEAN$"]
    cols = [c for c in tau_df.columns]
    for p in patterns:
        pat = re.compile(p)
        for c in cols:
            if pat.search(c.replace('-', '_')):
                return c
    # try contains region name
    for c in cols:
        if region.replace('_','').upper() in c.replace('_','').upper():
            if 'SUVR' in c.upper() or 'CTX' in c.upper():
                return c
    return None


def discover_tau_columns(tau_df: pd.DataFrame, canon: List[str] = CANON) -> Dict[str, Optional[str]]:
    res = {k: tau_col_for_key(tau_df, k) for k in canon}
    missing = [k for k,v in res.items() if v is None]
    if missing:
        print("Warning - missing tau columns for keys:", missing)
    return res


def filter_qc(fsx: pd.DataFrame, tau: pd.DataFrame, amy: pd.DataFrame):
    fsx_filt = fsx.copy()
    if 'OVERALLQC' in fsx_filt.columns:
        fsx_filt = fsx_filt[fsx_filt['OVERALLQC'].str.upper()=='PASS']
    # tau QC column may differ; leave as-is but user can filter externally
    return fsx_filt, tau, amy


def age_at_scan_merge(scan_df: pd.DataFrame, mrg: pd.DataFrame, datecol: str) -> pd.DataFrame:
    # Try to get age at scan from mrg if available; prefer AGE variables
    m = mrg.copy()
    m.columns = [c.upper() for c in m.columns]
    scan = scan_df.copy()
    scan.columns = [c.upper() for c in scan.columns]
    if 'RID' not in m.columns:
        return scan
    candidates = [c for c in m.columns if 'AGE' in c]
    if candidates:
        agecol = candidates[0]
        merged = scan.merge(m[['RID', agecol]], on='RID', how='left')
        merged = merged.rename(columns={agecol: 'AGE_SCAN'})
        return merged
    # fallback: no AGE available — leave AGE_SCAN NaN
    scan['AGE_SCAN'] = np.nan
    return scan


def collapse_duplicate_rids(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate RID rows by taking the first non-missing value per column."""
    def first_notna(series: pd.Series):
        for v in series:
            if pd.notna(v):
                return v
        return np.nan
    return df.groupby('RID', as_index=False).agg(first_notna)


def normalize_dx(series: pd.Series) -> pd.Series:
    """Normalize diagnosis strings to canonical categories."""
    def norm(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().upper()
        if s.startswith('DEM') or s == 'AD' or s == 'DEMENTIA':
            return 'Dementia'
        if s in ('MCI','EMCI','LMCI'):
            return s
        if s in ('CN','NORMAL'):
            return 'CN'
        if s in ('SMC',):
            return 'SMC'
        return s
    return series.map(norm)


def build_matrix(scan1: pd.DataFrame, region_cols: Dict[str,str], datecol: str,
                 case_mask: pd.Series, ctrl_mask: pd.Series, mrg: pd.DataFrame) -> pd.DataFrame:
    cols = ['RID'] + [c for c in region_cols.values() if c in scan1.columns]
    M = scan1[cols].merge(mrg[['RID']], on='RID', how='left')
    M = age_at_scan_merge(M, mrg, datecol)
    # attempt to get sex
    if 'PTGENDER' in mrg.columns:
        M = M.merge(mrg[['RID','PTGENDER']], on='RID', how='left')
    elif 'PTGENDER' in M.columns:
        pass
    else:
        M['PTGENDER'] = np.nan

    # Group assignment
    case_series = M['RID'].map(case_mask).fillna(False).astype(bool)
    ctrl_series = M['RID'].map(ctrl_mask).fillna(False).astype(bool)
    M['GRP'] = pd.Series(np.nan, index=M.index, dtype=object)
    M.loc[case_series, 'GRP'] = 'case'
    M.loc[~case_series & ctrl_series, 'GRP'] = 'ctrl'
    M = M.dropna(subset=['GRP']).reset_index(drop=True)
    # rename region columns to canonical keys
    inv = {v:k for k,v in region_cols.items()}
    rename_map = {c: inv[c] for c in cols if c in inv}
    M = M.rename(columns=rename_map)
    return M


def adjusted_d(M: pd.DataFrame, region_keys: List[str], higher_in_disease: bool=True, min_n:int=10) -> Dict[str, Optional[float]]:
    out = {}
    for r in region_keys:
        if r not in M.columns:
            out[r] = np.nan
            continue
        d = M[["GRP","AGE_SCAN","PTGENDER", r]].dropna()
        if d.shape[0] < min_n or d['GRP'].nunique() < 2:
            out[r] = np.nan
            continue
        covars = []
        if 'AGE_SCAN' in d.columns and d['AGE_SCAN'].notna().any():
            covars.append('AGE_SCAN')
        if 'PTGENDER' in d.columns and d['PTGENDER'].notna().any():
            covars.append('C(PTGENDER)')
        formula = f"{r} ~ " + (" + ".join(covars) if covars else "1")
        try:
            mod = smf.ols(formula, data=d).fit()
            resid = mod.resid
            grp = d['GRP']
            mean_case = resid[grp=='case'].mean()
            mean_ctrl = resid[grp=='ctrl'].mean()
            pooled_sd = resid.std(ddof=1)
            if pooled_sd == 0 or np.isnan(pooled_sd):
                out[r] = np.nan
            else:
                val = (mean_case - mean_ctrl) / pooled_sd
                if not higher_in_disease:
                    val = -val
                out[r] = float(val)
        except Exception:
            out[r] = np.nan
    return out


def run_pipeline(TAU: str, FSX: str, AMY: str, MRG: str, out_dir: str = "processed/pet"):
    os.makedirs(out_dir, exist_ok=True)
    tau, fsx, amy, mrg = load_csvs(TAU, FSX, AMY, MRG)
    tau, fsx, amy, mrg = map(tidy, (tau, fsx, amy, mrg))
    tau = dedup(tau, 'SCANDATE')
    fsx = dedup(fsx, 'EXAMDATE')
    amy = dedup(amy, 'SCANDATE')

    # Step 0: discover tau columns
    TAU_SUVR = discover_tau_columns(tau, CANON)

    # Step 3: QC
    fsx_filt, tau_filt, amy_filt = filter_qc(fsx, tau, amy)

    # Build grouping table from mrg: create boolean masks per RID
    mrg_up = mrg.copy()
    mrg_up.columns = [c.upper() for c in mrg_up.columns]
    if 'RID' not in mrg_up.columns:
        raise ValueError('MRG table must contain RID')
    if mrg_up['RID'].duplicated().any():
        mrg_up = collapse_duplicate_rids(mrg_up)
    # normalize DX labels
    if 'DX' in mrg_up.columns:
        mrg_up['DX_NORM'] = normalize_dx(mrg_up['DX'])
    else:
        mrg_up['DX_NORM'] = pd.Series(index=mrg_up.index)
    if 'DX_BL' in mrg_up.columns:
        mrg_up['DX_BL_NORM'] = normalize_dx(mrg_up['DX_BL'])
    else:
        mrg_up['DX_BL_NORM'] = pd.Series(index=mrg_up.index)

    # Basic amyloid positive mapping: if amy table has a SUVR or flag column try to infer
    # For now, build masks using mrg if ABETA_POS exists
    group = mrg_up[['RID']].drop_duplicates()
    if 'ABETA_POS' in mrg_up.columns:
        amy_status = mrg_up.set_index('RID')['ABETA_POS'].astype(bool)
    else:
        amy_status = pd.Series(False, index=group['RID'].values)

    # Build masks (lazy - user should refine per plan)
    # Use DX_BL and DX from mrg when available
    dx_bl = mrg_up.set_index('RID')['DX_BL_NORM'] if 'DX_BL_NORM' in mrg_up.columns else pd.Series(index=group['RID'].values)
    dx = mrg_up.set_index('RID')['DX_NORM'] if 'DX_NORM' in mrg_up.columns else pd.Series(index=group['RID'].values)

    # Define early_spectrum
    early_spectrum = dx_bl.isin(['CN','SMC','EMCI','LMCI','MCI']) if not dx_bl.empty else pd.Series(False, index=group['RID'].values)
    TAU_CASE = (amy_status == True) & early_spectrum
    TAU_CTRL = (amy_status == False) & (dx_bl == 'CN')
    ATR_CASE = dx == 'Dementia'
    ATR_CTRL = dx == 'CN'

    # Build matrices
    tau_M = build_matrix(tau_filt, TAU_SUVR, 'SCANDATE', TAU_CASE, TAU_CTRL, mrg_up)
    atr_M = build_matrix(fsx_filt, FS_TA, 'EXAMDATE', ATR_CASE, ATR_CTRL, mrg_up)

    # Compute per-region effect sizes
    tau_scores = adjusted_d(tau_M, [k for k in CANON], higher_in_disease=True)
    atr_scores = adjusted_d(atr_M, [k for k in CANON], higher_in_disease=False)

    tau_df = pd.DataFrame.from_dict(tau_scores, orient='index', columns=['effect_size']).rename_axis('region')
    atr_df = pd.DataFrame.from_dict(atr_scores, orient='index', columns=['effect_size']).rename_axis('region')

    tau_out = os.path.join(out_dir, 'tau_target.csv')
    atr_out = os.path.join(out_dir, 'atrophy_target.csv')
    tau_df.to_csv(tau_out)
    atr_df.to_csv(atr_out)
    print('Wrote', tau_out, atr_out)
    return tau_df, atr_df


if __name__ == '__main__':
    # example usage (won't run unless paths exist)
    print('pet_pipeline module loaded. Use run_pipeline(TAU, FSX, AMY, MRG, out_dir)')
