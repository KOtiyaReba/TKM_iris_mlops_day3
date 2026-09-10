from sklearn.model_selection import train_test_split
import pandas as pd
from config import RAW_DATA, PROCESSED_DIR, FEATURES, TARGET, RANDOM_STATE

def main():
    df = pd.read_csv(RAW_DATA)

    # do all the preprocessing steps here.
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    train[TARGET] = y_train.values
    test = X_test.copy()
    test[TARGET] = y_test.values

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print("Preprocessing complete")

if __name__ == "__main__":
    main()
