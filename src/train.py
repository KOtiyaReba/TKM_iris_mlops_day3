import mlflow
import json
import mlflow.sklearn

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import PROCESSED_DIR, MODEL_DIR, FEATURES, TARGET, RANDOM_STATE, REPORTS_DIR

def main():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    n_estimators = 100
    max_depth = 5

    mlflow.set_experiment('iris-classification')

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE
        )
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        accuracy = accuracy_score(test[TARGET], pred)
        print(f"accuracy={accuracy:.4f}")

        # -------------------------
        # MLflow tracking
        # -------------------------

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state",RANDOM_STATE)
        mlflow.log_metric("accuracy", accuracy)

        # -------------------------
        # Save model normally
        # -------------------------

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = (MODEL_DIR /"iris_random_forest.joblib")
        joblib.dump(model, model_path)

        # -------------------------
        # Log model into MLflow
        # -------------------------

        mlflow.sklearn.log_model(model, name="iris_model")

        # -------------------------
        # Save metric for DVC / CI
        # -------------------------

        REPORTS_DIR.mkdir(parents= True,exist_ok=True)

        with open(REPORTS_DIR/"metrics.json","w") as f:
            json.dump(
                {
                    "accuracy": accuracy
                },
                f,
                indent=4
            )

    print("Training completed successfully")


if __name__ == "__main__":
    main()