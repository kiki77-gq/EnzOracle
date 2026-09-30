"""Train the classification and two regression models sequentially with one seed."""

import subprocess
import sys
from pathlib import Path


# Change this value to train all three models with another reproducible seed.
SEED = 502

MODEL_ROOT = Path(__file__).resolve().parent
TM_ROOT = MODEL_ROOT.parent
DATA_ROOT = TM_ROOT / 'data'
EMBEDDING_DIR = TM_ROOT / 'embedding'
TRAIN_OUTPUT_ROOT = TM_ROOT / 'train_model'

TRAINING_JOBS = (
    (
        MODEL_ROOT / 'model_cls' / 'traincls.py',
        DATA_ROOT / 'Tm.csv',
    ),
    (
        MODEL_ROOT / 'model_reg5060' / 'trainreg.py',
        DATA_ROOT / 'tm_50_60.csv',
    ),
    (
        MODEL_ROOT / 'model_regrest' / 'trainreg.py',
        DATA_ROOT / 'tm_rest.csv',
    ),
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
