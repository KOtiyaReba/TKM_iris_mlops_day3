import pandas as pd
from config import FEATURES, TARGET, VALID_TARGETS

def validate(path):
    df = pd.read_csv(path)
    errors = []

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    for c in FEATURES:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.isna().sum() > df[c].isna().sum():
            errors.append(f"{c}: non-numeric values")
        if df[c].isna().any():
            errors.append(f"{c}: missing values")
        if (numeric.dropna() <= 0).any():
            errors.append(f"{c}: non-positive values")

    if not set(df[TARGET].dropna().unique()).issubset(VALID_TARGETS):
        errors.append("target: invalid labels")

    return errors