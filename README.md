# EnzOracle-Topt

## Folder Structure

```text
Topt/
├── EnzoracleTopt.py                # Standalone prediction script for user sequences
│
├── data/
│   ├── Topt.csv                     # Dataset used by the classification model
│   ├── topt_20_40.csv               # Dataset used by the 20–40 regression model
│   ├── topt_rest.csv                # Dataset used by the remaining-range regression model
│   └── data_dict.npy                # Amino-acid vocabulary
│
├── feature/
│   └── esm_feature.py               # ESM2 embedding extraction script
│
├── embedding/                       # ESM embeddings consumed by training (not versioned)
│
├── model/
│   ├── model_cls/                   # Classification model and training code
│   ├── model_reg2040/               # Regression model for topt_20_40.csv
│   ├── model_regrest/               # Regression model for topt_rest.csv
│   ├── modelTopt.py                 # Integrated model used for prediction
│   ├── modelToptfind.py             # Integrated model used during checkpoint selection
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
`Topt.csv` | Classification | `ID`, `topt`, `sequence`, `Split`, `weight`, `in_20_40`
`topt_20_40.csv` | Regression | `ID`, `topt`, `sequence`, `Split`, `weight`
`topt_rest.csv` | Regression | `ID`, `topt`, `sequence`, `Split`, `weight`

`ID` must be unique within the data used for training and must match the filename of its ESM embedding. For example, an entry with `ID = 1` requires the embedding file `embedding/1.npy`.

The `Split` column is used directly by the training and checkpoint-selection scripts. It must contain `Training`, `Validation`, and `Testing` for the corresponding samples. The `weight` column provides the sample weight used during optimization. In `Topt.csv`, `in_20_40` is the binary target used by the classification model.

The provided CSV files also contain metadata columns such as `uniprot`, `ogt`, and `ph`. These columns are retained in the distributed data files but are not read by the current training data loaders.

## ESM Feature Extraction

The models require one ESM representation for every sequence ID before training. The extraction script uses the `esm2_t33_650M_UR50D` model and reads sequences from `data/Topt.csv`.

From the `Topt/` directory, run:

```bash
python feature/esm_feature.py
```

Each representation is saved as an `.npy` file named after the corresponding `ID`.



## Train All Models

`model/train_all_models.py` is the single training entry point. One execution sequentially trains all three models with the same seed:

1. The classification model using `data/Topt.csv`.
2. The regression model using `data/topt_20_40.csv`.
3. The regression model using `data/topt_rest.csv`.

Run the following command from the `Topt/` directory:

```bash
python model/train_all_models.py
```

The seed is controlled by the `SEED` variable at the top of `model/train_all_models.py`. The selected seed is forwarded unchanged to all three training scripts. Each training run writes per-epoch checkpoints and a `log.csv` file required by the subsequent checkpoint-combination selection step.

## Training Outputs

All outputs are written under `train_model/`. With the default seed `502`, the three output directories are:

```text
train_model/
├── toptcls_seed502/
├── toptreg2040_seed502/
└── toptregrest_seed502/
```

Each directory contains the best checkpoint (`model_best.pt`), the most recent checkpoint (`model_recent.pt`), per-epoch checkpoints (`model_epoch_*.pt`), and the training log (`log.csv`).

## Select the Best Checkpoint Combination

After all three training jobs have finished, run `testfind.py` before final prediction.

For seed `502`, run from the `Topt/` directory:

```bash
python model/testfind.py \
  --cls_dir train_model/toptcls_seed502 \
  --reg1_dir train_model/toptreg2040_seed502 \
  --reg2_dir train_model/toptregrest_seed502
```

The results are written to:

```text
train_model/grid_search_validation_results_seed502.csv
```


`testfind.py` prints the selected `cls_epoch`, `reg1_epoch`, and `reg2_epoch`. Before standalone prediction, set `DEFAULT_CLS_CKPT`, `DEFAULT_REG1_CKPT`, and `DEFAULT_REG2_CKPT` in `model/modelTopt.py` to the corresponding `model_epoch_*.pt` files from the selected combination.



## Prediction for User Sequences

After selecting the checkpoint combination, predict Topt values for new sequences with the standalone `EnzoracleTopt.py` script from the `Topt/` directory:

```bash
python EnzoracleTopt.py --input path/to/sequences.fasta --output prediction_results.csv
```

This script uses the three selected epoch checkpoints configured in `model/modelTopt.py`. It generates ESM2 representations for new sequences during prediction and does not require a precomputed `embedding/` directory.

### Accepted Input Formats

Format | Requirements
--- | ---
FASTA (`.fasta` or `.fa`) | Use standard FASTA records: each sequence starts with a `>` identifier line followed by its amino-acid sequence.
CSV (`.csv`) | Provide a `sequence` column (or `Sequence`). The supplied example uses the header `ID,sequence`. An `ID` column is used when present; otherwise, the first column is used as the sequence identifier.
Text (`.txt`) | Put one amino-acid sequence on each non-empty line, without a header. Identifiers are assigned automatically as `Seq_1`, `Seq_2`, and so on.

For each input sequence, `EnzoracleTopt.py` first searches `data/Topt.csv` for an exact sequence match:

- Exact matches return the database Topt value with `Source = Database Exact Match`.
- Sequences not found in the database are predicted by the integrated model with `Source = AI Model Prediction`.

The output CSV contains `ID`, `Sequence`, `Topt`, and `Source`.

Optional arguments are available when non-default paths are needed:

```bash
python EnzoracleTopt.py \
  --input path/to/sequences.csv \
  --output prediction_results.csv \
  --db_path data/Topt.csv \
  --vocab_path data/data_dict.npy \
  --batch_size 8 \
  --seq_max_len 1024
```
