from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data/raw/iris.csv"
PROCESSED_DIR = ROOT / "data/processed"
MODEL_DIR = ROOT / "models"

QUARANTINE_DIR = ROOT / "data/quarantine"
REPORTS_DIR = ROOT / "reports"

FEATURES = ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]
TARGET = "target"

RANDOM_STATE = 42
VALID_TARGETS = {0,1,2}
