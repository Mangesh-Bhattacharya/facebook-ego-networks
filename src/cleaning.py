import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED
from src.data_loading import load_raw_data


def clean_facebook_dataset(raw_path: str = DATA_RAW, save: bool = True) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    before = len(df)

    df[["node1", "node2"]] = pd.DataFrame(
        df.apply(lambda r: sorted([r["node1"], r["node2"]]), axis=1).tolist()
    )
    df = df.drop_duplicates()
    df = df[df["node1"] != df["node2"]]
    df = df.reset_index(drop=True)

    after = len(df)
    print(f"  Cleaning : {before:,} raw rows → {after:,} clean edges ({before - after:,} removed)")

    if save:
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        out = os.path.join(DATA_PROCESSED, "edges_clean.csv")
        df.to_csv(out, index=False)
        print(f"  Saved    : {out}")

    return df


def run():
    print()
    print("=" * 50)
    print("  Phase 1 — Data Cleaning & Overview")
    print("=" * 50)

    clean_edges = clean_facebook_dataset(raw_path=DATA_RAW, save=True)

    all_nodes = pd.concat([clean_edges["node1"], clean_edges["node2"]]).unique()

    print()
    print("  Dataset Overview")
    print(f"  {'Total edges':<22}: {len(clean_edges):,}")
    print(f"  {'Total unique nodes':<22}: {len(all_nodes):,}")
    print(f"  {'Node ID range':<22}: {all_nodes.min()} → {all_nodes.max()}")

    print()
    print("  First 10 edges:")
    print(clean_edges.head(10).to_string(index=False))

    degree_freq = (
        pd.concat([clean_edges["node1"], clean_edges["node2"]])
        .value_counts()
        .rename_axis("Node")
        .reset_index(name="Degree")
    )

    print()
    print("  Top 10 highest-degree nodes:")
    print(degree_freq.head(10).to_string(index=False))

    out = os.path.join(DATA_PROCESSED, "degree_frequency.csv")
    degree_freq.to_csv(out, index=False)
    print()
    print(f"  Saved degree frequency → {out}")
    print()
    print("  Phase 1 complete.")
    print()

    return clean_edges


if __name__ == "__main__":
    run()
