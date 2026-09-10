# iris_mlops
This demonstrates how MLOps work using Iris dataset and ML models created using it.

# Build the Project

Goal: establish a production-style ML repository before introducing automation.

Flow:
raw data -> preprocess -> train -> model artifact

Run:
```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
```
# Data Engineering

Industry-style flow:

incoming batch
    -> validate
    -> reject to quarantine OR promote to raw/iris.csv
    -> preprocess
    -> train

The ingestion job is the entry point:

```bash
python src/ingest_batch.py data/incoming/iris_batch_good.csv
```

Bad data:
```bash
python src/ingest_batch.py data/incoming/iris_batch_bad.csv
```
# DVC
Industry-style behavior:
new approved data
    -> promoted to data/raw/iris.csv
    -> DVC detects changed dependency
    -> dvc repro reruns only invalidated stages
    -> preprocessing reruns
    -> training reruns
    -> model is regenerated
Local trigger:
```bash
pip install dvc
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
# test preprocess.py and train.py
# DVC and Git should not track the same file
# so, output files moed out of git tracking
git rm --cached data/processed/train.csv
git rm --cached data/processed/test.csv
git rm --cached models/iris_random_forest.joblib

# prepare yaml script
dvc stage add -n preprocess -d data/raw/iris.csv -d src/preprocess.py -d src/config.py -o data/processed/train.csv -o data/processed/test.csv "python src/preprocess.py"
dvc stage add -n train -d data/processed/train.csv -d data/processed/test.csv -d src/train.py -d src/config.py -o models/iris_random_forest.joblib "python src/train.py"
# verify DVC creation
# test DVC yaml 
dvc repro
# run 'dvc repro' again to see no change = no execution

python src/ingest_batch.py data/incoming/iris_v2.csv