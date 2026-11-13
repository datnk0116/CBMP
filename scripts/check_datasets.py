#!/usr/bin/env python3
"""
scripts/check_datasets.py

Validate that all datasets referenced in an experiment config (e.g. experiments/configs.yaml)
exist on disk in the dataset directory (e.g. datasets/).
Reports a clear error message and lists all missing datasets if any are missing.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

def find_referenced_datasets(config_path, groups=None):
    """Extract dataset names from all groups or a selected subset."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    referenced = set()
    config_groups = config.get("groups", {})
    selected_names = None if groups is None else set(groups)
    selected = config_groups.items() if selected_names is None else ((name, cfg) for name, cfg in config_groups.items() if name in selected_names)
    for grp_name, grp_cfg in selected:
        ds = grp_cfg.get("datasets")
        if isinstance(ds, list):
            for d in ds:
                if isinstance(d, str) and d != "all":
                    referenced.add(d)
                elif isinstance(d, dict) and "name" in d:
                    referenced.add(d["name"])
        elif isinstance(ds, dict):
            for d_name in ds.keys():
                referenced.add(d_name)

    return sorted(list(referenced))

def validate_datasets(config_path="experiments/configs.yaml", dataset_dir="datasets", groups=None, dataset_names=None):
    """
    Check all referenced datasets in config against dataset_dir.
    Returns:
        found (list of str): List of found dataset names.
    Raises:
        FileNotFoundError if any dataset referenced in config does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    referenced = sorted(set(dataset_names)) if dataset_names is not None else find_referenced_datasets(config_path, groups=groups)
    missing = []
    found = []

    for name in referenced:
        file_path = os.path.join(dataset_dir, f"{name}.mtx" if not name.endswith(".mtx") else name)
        if os.path.isfile(file_path):
            found.append(name)
        else:
            missing.append(name)

    if missing:
        msg = (
            f"ERROR: {len(missing)} dataset(s) referenced in '{config_path}' were NOT found in '{dataset_dir}/':\n"
            + "\n".join(f"  - {m} (expected file: {os.path.join(dataset_dir, f'{m}.mtx')})" for m in missing)
            + f"\nPlease add the missing .mtx file(s) to '{dataset_dir}/'."
        )
        raise FileNotFoundError(msg)

    return found

def main():
    parser = argparse.ArgumentParser(description="Verify that all datasets referenced in config exist on disk.")
    parser.add_argument("--config", default="experiments/configs.yaml", help="Path to experiment config YAML")
    parser.add_argument("--dataset-dir", default="datasets", help="Directory containing dataset files")
    args = parser.parse_args()

    try:
        found = validate_datasets(config_path=args.config, dataset_dir=args.dataset_dir)
        print(f"[OK] All {len(found)} datasets referenced in '{args.config}' exist in '{args.dataset_dir}/': {', '.join(found)}")
        sys.exit(0)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

