"""
cleaning.py
Raw-data cleaning utilities and Phase-1 pipeline runner.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED
from src.data_loading import load_raw_data


def clean_facebook_dataset(raw_path: str = DATA_RAW,
                            save: bool = True) -> pd.DataFrame:
    """
    Load and clean the raw Facebook edge list.

    Steps
    -----
    1. Load with auto-format detection (see data_loading.py).
    2. Drop duplicate edges (both orderings count as the same edge).
    3. Remove self-loops.
    4. Reset index.
    5. Optionally save to data/processed/edges_clean.csv.

    Returns
    -------
    pd.DataFrame with columns ['node1', 'node2'].
    """
    df = load_raw_data(raw_path)

    before = len(df)

    # Normalise edge direction so (u,v) and (v,u) are treated identically
    df[["node1", "node2"]] = pd.DataFrame(
        df.apply(lambda r: sorted([r["node1"], r["node2"]]), axis=1).tolist()
    )

    # Drop duplicates and self-loops
    df = df.drop_duplicates()
    df = df[df["node1"] != df["node2"]]
    df = df.reset_index(drop=True)

    after = len(df)
    removed = before - after
    print(f"  Cleaning : {before:,} raw rows → {after:,} clean edges " f"({removed:,} removed)")

    if save:
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        out = os.path.join(DATA_PROCESSED, "edges_clean.csv")
        df.to_csv(out, index=False)
        print(f"  Saved clean edges → {out}")

    return df


# ── Phase-1 pipeline runner ────────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  PHASE 1 — Data Cleaning & Overview")
    print("=" * 55)

    # ── Step 1: Clean raw dataset ──────────────────────────────
    clean_edges = clean_facebook_dataset(raw_path=DATA_RAW, save=True)

    # ── Step 2: Quick overview ─────────────────────────────────
    print("\n  Dataset Overview")
    print(f"  Total edges  : {len(clean_edges):,}")
    print(f"  Unique node1 : {clean_edges['node1'].nunique():,}")
    print(f"  Unique node2 : {clean_edges['node2'].nunique():,}")

    all_nodes = pd.concat([clean_edges["node1"], clean_edges["node2"]]).unique()
    print(f"  Total unique nodes : {len(all_nodes):,}")
    print(f"  Node ID range      : {all_nodes.min()} → {all_nodes.max()}")

    # ── Step 3: Sample rows ────────────────────────────────────
    print("\n  First 10 edges:")
    print(clean_edges.head(10).to_string(index=False))

    # ── Step 4: Degree frequency preview ──────────────────────
    degree_freq = (
        pd.concat([clean_edges["node1"], clean_edges["node2"]])
        .value_counts()
        .rename_axis("Node")
        .reset_index(name="Degree")
    )
    print("\n  Top 10 highest-degree nodes:")
    print(degree_freq.head(10).to_string(index=False))

    # ── Step 5: Save degree frequency ─────────────────────────
    out = os.path.join(DATA_PROCESSED, "degree_frequency.csv")
    degree_freq.to_csv(out, index=False)
    print(f"\n  Saved degree frequency → {out}")

    print("\n  Phase 1 complete.\n")
    return clean_edges


if __name__ == "__main__":
    run()
