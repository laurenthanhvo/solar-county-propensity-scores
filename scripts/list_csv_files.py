#!/usr/bin/env python3
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent

    csv_paths = sorted(
        p.relative_to(repo_root)
        for p in repo_root.rglob("*.csv")
        if ".ipynb_checkpoints" not in str(p)
    )

    print(f"Found {len(csv_paths)} CSV files:\n")
    for p in csv_paths:
        print(p)

if __name__ == "__main__":
    main()