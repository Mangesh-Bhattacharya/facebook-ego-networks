import pandas as pd
import os

def load_raw_data(path: str) -> pd.DataFrame:
    # Peek at the first non-comment line to detect format
    with open(path, "r") as fh:
        first_line = fh.readline().strip()

    sep = "," if "," in first_line else r"\s+"

    # Detect whether there is a text header row
    try:
        int(first_line.split("," if sep == "," else " ")[0])
        header = None  # first token is numeric → no header
    except ValueError:
        header = 0     # first token is text → has header

    df = pd.read_csv(path, sep=sep, header=header, engine="python", comment="#")
    df.columns = ["node1", "node2"]
    df = df.astype({"node1": int, "node2": int})
    print(f"  Loaded raw data : {len(df):,} rows from {os.path.basename(path)}")
    return df

def load_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.astype({"node1": int, "node2": int})
    print(f"  Loaded clean data : {len(df):,} edges from {os.path.basename(path)}")
    return df
