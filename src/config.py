import os

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw", "facebook_combined.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
FIGURES_PATH   = os.path.join(BASE_DIR, "figures")
LOGS_PATH      = os.path.join(BASE_DIR, "logs")
REPORTS_PATH   = os.path.join(BASE_DIR, "reports")

EGO_NODE    = 0     # Central node for ego-network analysis
K_CLIQUE    = 3     # k value for k-clique percolation
BA_M        = 4     # Edges per new node in Barabasi-Albert model
RANDOM_SEED = 42
DIFFUSION_P = 0.1   # Propagation probability for Independent Cascade
