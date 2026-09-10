import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config import PROCESSED_DIR, MODEL_DIR, FEATURES, TARGET, RANDOM_STATE

def main():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])

    # create multiple models, compare and select the best model

    print(f"accuracy={accuracy_score(test[TARGET], pred):.4f}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "iris_random_forest.joblib")

if __name__ == "__main__":
    main()