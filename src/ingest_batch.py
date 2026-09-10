import shutil
import subprocess
import sys
from pathlib import Path
from validate_data import validate
from config import RAW_DATA, QUARANTINE_DIR

def main(batch_path):
    batch = Path(batch_path)
    errors = validate(batch)

    if errors:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        dest = QUARANTINE_DIR / batch.name
        shutil.copy2(batch, dest)
        print("BATCH REJECTED")
        for e in errors:
            print(" -", e)
        print("Quarantined at:", dest)
        raise SystemExit(1)

    shutil.copy2(batch, RAW_DATA)
    print("BATCH APPROVED AND PROMOTED:", RAW_DATA)

    # Industry pattern: ingestion success triggers downstream pipeline automatically.
    subprocess.run([sys.executable, "src/preprocess.py"], check=True)
    subprocess.run([sys.executable, "src/train.py"], check=True)

if __name__ == "__main__":
    main(sys.argv[1])