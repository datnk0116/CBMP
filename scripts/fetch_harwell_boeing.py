#!/usr/bin/env python3
"""
scripts/fetch_harwell_boeing.py

Download the 28 Harwell-Boeing benchmark instances referenced in NILS Table A.2
from NIST MatrixMarket (preserving existing files cleanly without modifying graph data),
save raw `.mtx.gz` archives to `datasets/raw/` and decompressed `.mtx` files to `datasets/`,
and verify parsed (N, |E|) (after removing self-loops, symmetrizing, and deduplicating edges)
against the reference dimensions in Table B2.
"""

import argparse
import gzip
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import issparse

BASE_URL = "https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/"

# Ordered list of (canonical_name, remote_subpath, expected_N, expected_E)
HARWELL_BOEING_28 = [
    ("jgl009", "counterx/jgl009.mtx.gz", 9, 50),
    ("rgg010", "counterx/rgg010.mtx.gz", 10, 76),
    ("jgl011", "counterx/jgl011.mtx.gz", 11, 76),
    ("can_24", "cannes/can___24.mtx.gz", 24, 92),
    ("pores_1", "pores/pores_1.mtx.gz", 30, 103),
    ("ibm32", "smtape/ibm32.mtx.gz", 32, 90),
    ("bcspwr01", "bcspwr/bcspwr01.mtx.gz", 39, 46),
    ("bcsstk01", "bcsstruc1/bcsstk01.mtx.gz", 48, 176),
    ("bcspwr02", "bcspwr/bcspwr02.mtx.gz", 49, 59),
    ("curtis54", "smtape/curtis54.mtx.gz", 54, 124),
    ("will57", "smtape/will57.mtx.gz", 57, 127),
    ("impcol_b", "chemimp/impcol_b.mtx.gz", 59, 281),
    ("ash85", "smtape/ash85.mtx.gz", 85, 219),
    ("nos4", "lanpro/nos4.mtx.gz", 100, 247),
    ("dwt_234", "dwt/dwt__234.mtx.gz", 117, 162),
    ("bcspwr03", "bcspwr/bcspwr03.mtx.gz", 118, 179),
    ("bcsstk06", "bcsstruc1/bcsstk06.mtx.gz", 420, 3720),
    ("bcsstk07", "bcsstruc1/bcsstk07.mtx.gz", 420, 3720),
    ("impcol_d", "chemimp/impcol_d.mtx.gz", 425, 1267),
    ("can_445", "cannes/can__445.mtx.gz", 445, 1682),
    ("494_bus", "psadmit/494_bus.mtx.gz", 494, 586),
    ("dwt_503", "dwt/dwt__503.mtx.gz", 503, 2762),
    ("sherman4", "sherman/sherman4.mtx.gz", 546, 1341),
    ("dwt_592", "dwt/dwt__592.mtx.gz", 592, 2256),
    ("662_bus", "psadmit/662_bus.mtx.gz", 662, 906),
    ("nos6", "lanpro/nos6.mtx.gz", 675, 1290),
    ("685_bus", "psadmit/685_bus.mtx.gz", 685, 1282),
    ("can_715", "cannes/can__715.mtx.gz", 715, 2975),
]


def parse_undirected_dimensions(mtx_path: str):
    """
    Parse a MatrixMarket file using scipy.io.mmread and return:
      (N, num_undirected_edges, header_symmetric_flag, structurally_symmetric)
    after removing self-loops, symmetrizing, and deduplicating edges.
    Does NOT modify the file on disk.
    """
    header_sym = "unknown"
    with open(mtx_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
        if first_line.startswith("%%MatrixMarket"):
            parts = first_line.split()
            if len(parts) >= 5:
                header_sym = parts[4].lower()

    try:
        mat = mmread(mtx_path, spmatrix=True)
    except TypeError:
        mat = mmread(mtx_path)

    if not hasattr(mat, "shape") or len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{mtx_path} is not a square 2D matrix (shape={getattr(mat, 'shape', None)})")

    n = int(mat.shape[0])
    if issparse(mat):
        csr = mat.tocsr()
        rows, cols = csr.nonzero()
    else:
        rows, cols = np.nonzero(mat)

    mask = rows != cols
    r_nd = rows[mask]
    c_nd = cols[mask]

    directed_pairs = set(zip(r_nd.tolist(), c_nd.tolist()))
    structurally_symmetric = all((v, u) in directed_pairs for (u, v) in directed_pairs)
    undirected_edges = {(min(u, v), max(u, v)) for (u, v) in directed_pairs}
    return n, len(undirected_edges), header_sym, structurally_symmetric


def download_file(url: str, dest_path: Path, timeout: float = 30.0):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, dest_path)


def print_manual_instructions(missing_items):
    print("\n[OFFLINE / DOWNLOAD FAILED] Manual download instructions:", file=sys.stderr)
    print("Please download the following `.mtx.gz` files, place them in `datasets/raw/`", file=sys.stderr)
    print("(or decompress as `datasets/<instance>.mtx`), then re-run this script:\n", file=sys.stderr)
    for name, subpath, _, _ in missing_items:
        print(f"  - {name}: {BASE_URL}{subpath}  ->  datasets/raw/{name}.mtx.gz", file=sys.stderr)


def fetch_and_verify(dataset_dir: Path, raw_dir: Path, verify_only: bool = False):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    failed_downloads = []
    downloaded = []
    skipped_existing = []

    for name, subpath, _, _ in HARWELL_BOEING_28:
        mtx_path = dataset_dir / f"{name}.mtx"
        gz_path = raw_dir / f"{name}.mtx.gz"
        url = f"{BASE_URL}{subpath}"

        if mtx_path.exists():
            skipped_existing.append(name)
            continue

        if gz_path.exists():
            with gzip.open(gz_path, "rb") as fin, open(mtx_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            downloaded.append(name)
            continue

        if verify_only:
            failed_downloads.append((name, subpath, None, "missing on disk (--verify-only)"))
            continue

        try:
            download_file(url, gz_path)
            with gzip.open(gz_path, "rb") as fin, open(mtx_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            downloaded.append(name)
        except Exception as e:
            failed_downloads.append((name, subpath, url, str(e)))

    if failed_downloads:
        print_manual_instructions(failed_downloads)
        return 1

    rows = []
    mismatches = []
    asymmetric_files = []

    for name, subpath, exp_n, exp_e in HARWELL_BOEING_28:
        mtx_path = dataset_dir / f"{name}.mtx"
        n, e, hdr_sym, struct_sym = parse_undirected_dimensions(str(mtx_path))
        match = (n == exp_n) and (e == exp_e)
        if not match:
            mismatches.append((name, exp_n, exp_e, n, e))
        if not struct_sym:
            asymmetric_files.append((name, hdr_sym))
        rows.append((name, exp_n, exp_e, n, e, hdr_sym, struct_sym, match))

    print(f"Fetched newly: {len(downloaded)} | Already present: {len(skipped_existing)} | Total HB28: {len(rows)}")
    print("\n| Instance | Expected N | Expected |E| | Parsed N | Parsed |E| | MM Header | SymStruct | Match |")
    print("|:---|---:|---:|---:|---:|:---:|:---:|:---:|")
    for name, exp_n, exp_e, n, e, hdr_sym, struct_sym, match in rows:
        print(f"| `{name}` | {exp_n} | {exp_e} | {n} | {e} | {hdr_sym} | {'yes' if struct_sym else 'no (symmetrized)'} | {'OK' if match else 'MISMATCH'} |")

    if asymmetric_files:
        print(f"\nNote: {len(asymmetric_files)} matrix/matrices have non-symmetric directed entries (counted after symmetrization, files left unmodified):")
        for name, hdr in asymmetric_files:
            print(f"  - {name} (MatrixMarket header: {hdr})")

    if mismatches:
        print(f"\nWARNING: {len(mismatches)} dimension mismatch(es) detected (raw files NOT modified):")
        for name, exp_n, exp_e, n, e in mismatches:
            print(f"  - {name}: expected ({exp_n}, {exp_e}), got ({n}, {e})")
    else:
        print("\nAll 28 Harwell-Boeing instances match expected (N, |E|) exactly.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify the 28 Harwell-Boeing CBMP benchmark instances.")
    parser.add_argument("--dataset-dir", default="datasets", help="Directory for extracted .mtx files")
    parser.add_argument("--raw-dir", default="datasets/raw", help="Directory for raw downloaded .mtx.gz files")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files without downloading")
    args = parser.parse_args()
    sys.exit(fetch_and_verify(Path(args.dataset_dir), Path(args.raw_dir), verify_only=args.verify_only))


if __name__ == "__main__":
    main()

