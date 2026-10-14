# EnzOracle-pHopt

## Folder Structure

```text
pHopt/
├── EnzoraclepHopt.py               # Standalone prediction script for user sequences
│
├── data/
│   ├── pHopt.csv                    # Dataset used by the classification model
│   ├── pHopt_7_8.csv                # Dataset used by the pH 7–8 regression model
│   ├── pHopt_rest.csv               # Dataset used by the remaining-range regression model
│   └── data_dict.npy                # Amino-acid vocabulary
│
├── example/                         # Example inputs and prediction output
│   ├── test.csv
│   ├── test.fasta
│   ├── test.txt
│   └── prediction_results.csv
│
├── feature/
│   └── esm_feature.py               # ESM2 embedding extraction script
│
├── embedding/                       # ESM embeddings consumed by training (not versioned)
│
├── model/
│   ├── dataset.py                    # Shared data-loading utilities
│   ├── model_cls/                   # Classification model and training code
│   ├── model_reg78/                 # Regression model for pHopt_7_8.csv
│   ├── model_regrest/               # Regression model for pHopt_rest.csv
│   ├── modelpHopt.py                # Integrated model used for prediction
│   ├── modelpHoptfind.py            # Integrated model used during checkpoint selection
│   ├── test.py                      # Integrated-model evaluation script
│   ├── testfind.py                  # Checkpoint-combination selection script
│   └── train_all_models.py          # Training helper script
│
├── train_model/                     # Generated training outputs
└── README.md
```

## Data

The training entry point uses three CSV files in `data/`:

File | Training task | Required columns used by the training code
--- | --- | ---
`pHopt.csv` | Classification | `ID`, `pHopt`, `sequence`, `Split`, `weight`, `in_7_8`
`pHopt_7_8.csv` | Regression | `ID`, `pHopt`, `sequence`, `Split`, `weight`
`pHopt_rest.csv` | Regression | `ID`, `pHopt`, `sequence`, `Split`, `weight`

`ID` must be unique within the data used for training and must match the filename of its ESM embedding. For example, an entry with `ID = 1` requires the embedding file `embedding/1.npy`.

The `Split` column is used directly by the training and checkpoint-selection scripts. It must contain `Training`, `Validation`, and `Testing` for the corresponding samples. The `weight` column provides the sample weight used during optimization. In `pHopt.csv`, `in_7_8` is the binary target used by the classification model.

The provided CSV files also contain metadata columns such as `Accession`, `Organism`, `EC Number`, `Sample Weight`, and `length`. These columns are retained in the distributed data files but are not read by the current training data loaders.

## ESM Feature Extraction

The models require one ESM representation for every sequence ID before training. The extraction script uses the `esm2_t33_650M_UR50D` model and reads sequences from `data/pHopt.csv`.

From the `pHopt/` directory, run:

```bash
python feature/esm_feature.py
```

Each representation is saved as an `.npy` file named after the corresponding `ID`.



## Train All Models

`model/train_all_models.py` is the single training entry point. One execution sequentially trains all three models with the same seed:

1. The classification model using `data/pHopt.csv`.
2. The regression model using `data/pHopt_7_8.csv`.
3. The regression model using `data/pHopt_rest.csv`.

Run the following command from the `pHopt/` directory:

```bash
python model/train_all_models.py
```

The seed is controlled by the `SEED` variable at the top of `model/train_all_models.py` (default: `502`). The selected seed is forwarded unchanged to all three training scripts. Each training run writes per-epoch checkpoints and a `log.csv` file required by the subsequent checkpoint-combination selection step.

## Training Outputs

All outputs are written under `train_model/`. With the default seed `502`, the three output directories are:

```text
train_model/
├── phcls_seed502/
├── phreg78_seed502/
└── phregrest_seed502/
```

Each directory contains the best checkpoint (`model_best.pt`), the most recent checkpoint (`model_recent.pt`), per-epoch checkpoints (`model_epoch_*.pt`), and the training log (`log.csv`).

## Select the Best Checkpoint Combination

After all three training jobs have finished, run `testfind.py` before final prediction.

For seed `502`, run from the `pHopt/` directory:

```bash
python model/testfind.py \
  --cls_dir train_model/phcls_seed502 \
  --reg1_dir train_model/phreg78_seed502 \
  --reg2_dir train_model/phregrest_seed502
```

The results are written to:

```text
train_model/grid_search_validation_results_seed502.csv
```


`testfind.py` prints the selected `cls_epoch`, `reg1_epoch`, and `reg2_epoch`. Before standalone prediction, set `DEFAULT_CLS_CKPT`, `DEFAULT_REG1_CKPT`, and `DEFAULT_REG2_CKPT` in `model/modelpHopt.py` to the corresponding `model_epoch_*.pt` files from the selected combination.



## Prediction for User Sequences

After selecting the checkpoint combination, predict pHopt values for new sequences with the standalone `EnzoraclepHopt.py` script from the `pHopt/` directory:

```bash
python EnzoraclepHopt.py --input path/to/sequences.fasta --output prediction_results.csv
```

This script uses the three selected epoch checkpoints configured in `model/modelpHopt.py`. It generates ESM2 representations for new sequences during prediction and does not require a precomputed `embedding/` directory.

### Accepted Input Formats

Format | Requirements
--- | ---
FASTA (`.fasta` or `.fa`) | Use standard FASTA records: each sequence starts with a `>` identifier line followed by its amino-acid sequence.
CSV (`.csv`) | Provide a `sequence` column (or `Sequence`). The supplied example uses the header `ID,sequence`. An `ID` column is used when present; otherwise, the first column is used as the sequence identifier.
Text (`.txt`) | Put one amino-acid sequence on each non-empty line, without a header. Identifiers are assigned automatically as `Seq_1`, `Seq_2`, and so on.

For each input sequence, `EnzoraclepHopt.py` first searches `data/pHopt.csv` for an exact sequence match:

- Exact matches return the database pHopt value with `Source = Database Exact Match`.
- Sequences not found in the database are predicted by the integrated model with `Source = AI Model Prediction`.

The output CSV contains `ID`, `Sequence`, `pHopt`, and `Source`.

Optional arguments are available when non-default paths are needed:

```bash
python EnzoraclepHopt.py \
  --input path/to/sequences.csv \
  --output prediction_results.csv \
  --db_path data/pHopt.csv \
  --vocab_path data/data_dict.npy \
  --batch_size 8 \
  --seq_max_len 1024
```
