"""Orchestrate the full Part-1 pipeline and produce data/processed/*.npy.

Run from the project root with the venv active:
    python scripts/01_build_inputs.py

All raw files must be present in data/raw/ before running:
    data/raw/budapest_83.graphml
    data/raw/ftd_bvFTD_tmap.nii.gz
    data/raw/adni/  (UCSFFSX, DXSUM, MMSE CSVs)
"""

import sys
from pathlib import Path

# Allow `import levers` without an editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from levers.data import build_connectome, build_atrophy_ftd, build_atrophy_ad

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    print("=== Step 1/3  Connectome (Section 4b) ===")
    W, region_names = build_connectome(
        graphml_path=RAW / "budapest_83.graphml",
        out_dir=PROCESSED,
    )
    print(f"  W_dk68       shape={W.shape}  sum-per-row ~{W.sum(axis=1).mean():.4f}")
    print(f"  region_names n={len(region_names)}")

    print("\n=== Step 2/3  FTD atrophy map (Section 5b) ===")
    atrophy_ftd = build_atrophy_ftd(
        nifti_path=RAW / "ftd_bvFTD_tmap.nii.gz",
        region_names=region_names,
        out_dir=PROCESSED,
    )
    print(f"  atrophy_FTD  shape={atrophy_ftd.shape}  "
          f"min={atrophy_ftd.min():.3f}  max={atrophy_ftd.max():.3f}")

    print("\n=== Step 3/3  AD atrophy map (Section 5c) ===")
    atrophy_ad = build_atrophy_ad(
        adni_dir=RAW / "adni",
        region_names=region_names,
        out_dir=PROCESSED,
    )
    print(f"  atrophy_AD   shape={atrophy_ad.shape}  "
          f"min={atrophy_ad.min():.3f}  max={atrophy_ad.max():.3f}")

    print("\nAll Part-1 outputs written to data/processed/")
    print("  W_dk68.npy, region_names.npy, atrophy_AD.npy, atrophy_FTD.npy")


if __name__ == "__main__":
    main()
