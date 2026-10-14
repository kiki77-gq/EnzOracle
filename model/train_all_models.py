"""Train the pHopt classification and two regression models sequentially."""

import subprocess
import sys
from pathlib import Path


# Change this value to train all three models with another reproducible seed.
SEED = 502

MODEL_ROOT = Path(__file__).resolve().parent
PHOPT_ROOT = MODEL_ROOT.parent
DATA_ROOT = PHOPT_ROOT / 'data'
EMBEDDING_DIR = PHOPT_ROOT / 'embedding'
TRAIN_OUTPUT_ROOT = PHOPT_ROOT / 'train_model'

TRAINING_JOBS = (
    (MODEL_ROOT / 'model_cls' / 'traincls.py', DATA_ROOT / 'pHopt.csv'),
    (MODEL_ROOT / 'model_reg78' / 'trainreg.py', DATA_ROOT / 'pHopt_7_8.csv'),
    (MODEL_ROOT / 'model_regrest' / 'trainreg.py', DATA_ROOT / 'pHopt_rest.csv'),
)


def main():
    for train_script, csv_path in TRAINING_JOBS:
        command = [
            sys.executable,
            str(train_script),
            '--seed',
            str(SEED),
            '--csv_path',
            str(csv_path),
            '--esm_path',
            str(EMBEDDING_DIR),
            '--save_dir',
            str(TRAIN_OUTPUT_ROOT),
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, cwd=train_script.parent, check=True)


if __name__ == '__main__':
    main()
