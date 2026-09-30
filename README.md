## Folder Structure

```text
Tm/
├── EnzoracleTm.py                  # Standalone prediction script for user sequences
│
├── data/
│   ├── Tm.csv                       # Dataset used by the classification model
│   ├── tm_50_60.csv                 # Dataset used by the 50–60 regression model
│   ├── tm_rest.csv                  # Dataset used by the remaining-range regression model
│   └── data_dict.npy                # Amino-acid vocabulary
│
├── feature/
│   └── esm_feature.py               # ESM2 embedding extraction script
│
├── embedding/                       # ESM embeddings consumed by training (not versioned)
│
├── model/
│   ├── model_cls/                   # Classification model and training code
│   ├── model_reg5060/               # Regression model for tm_50_60.csv
│   ├── model_regrest/               # Regression model for tm_rest.csv
│   └── train_all_models.py          # Sequential training entry point
│
├── train_model/                     # Generated training outputs
└── README.md
```

## Data

The training entry point uses three CSV files in `data/`:

File | Training task | Required columns used by the training code
--- | --- | ---
`Tm.csv` | Classification | `ID`, `tm`, `sequence`, `Split`, `weight`, `in_50_60`
`tm_50_60.csv` | Regression | `ID`, `tm`, `sequence`, `Split`, `weight`
`tm_rest.csv` | Regression | `ID`, `tm`, `sequence`, `Split`, `weight`

`ID` must be unique within the data used for training and must match the filename of its ESM embedding. For example, an entry with `ID = 1` requires the embedding file `embedding/1.npy`.

The `Split` column is used directly by the training scripts. It must contain the values `Training` and `Validation` for the corresponding samples. The `weight` column provides the sample weight used during optimization. In `Tm.csv`, `in_50_60` is the binary target used by the classification model.

The provided CSV files also contain metadata columns such as `uniprot`, `ogt`, and `ph`. These columns are retained in the distributed data files but are not read by the current training data loaders.

## ESM Feature Extraction

The models require one ESM representation for every sequence ID before training. The extraction script uses the `esm2_t33_650M_UR50D` model and reads sequences from `data/Tm.csv`.

From the `Tm/` directory, run:

```bash
python feature/esm_feature.py
```

Each representation is saved as an `.npy` file named after the corresponding `ID`.



## Train All Models

`model/train_all_models.py` is the single training entry point. One execution sequentially trains all three models with the same seed:

1. The classification model using `data/Tm.csv`.
2. The regression model using `data/tm_50_60.csv`.
3. The regression model using `data/tm_rest.csv`.

Run the following command from the `Tm/` directory:

```bash
python model/train_all_models.py
```

The seed is controlled by the `SEED` variable at the top of `model/train_all_models.py`. The selected seed is forwarded unchanged to all three model-training scripts.

## Training Outputs

All outputs are written under `train_model/`. With the default seed `502`, the three output directories are:

```text
train_model/
├── tmcls_seed502/
├── tmreg5060_seed502/
└── tmregrest_seed502/
```

Each directory contains the best checkpoint (`model_best.pt`), the most recent checkpoint (`model_recent.pt`), per-epoch checkpoints (`model_epoch_*.pt`), and the training log (`log.csv`).

## Final Evaluation After Training

After all three training jobs have finished, run the integrated evaluation script to obtain final predictions for the held-out test set:

```bash
python model/test.py
```

The script reads samples with `Split == "Testing"` from `data/Tm.csv`, loads their precomputed ESM embeddings from `embedding/`, and combines the three trained sub-models through `FinalModel`.

With the published default seed (`502`), the script loads the following checkpoints:

```text
train_model/tmcls_seed502/model_best.pt
train_model/tmreg5060_seed502/model_best.pt
train_model/tmregrest_seed502/model_best.pt
```

The final evaluation writes two files to `train_model/`:

File | Description
--- | ---
`EnzOracle_502.csv` | Per-sample results with `ID`, experimental `tm`, final `prediction`, classification probability (`cls_prob`), and the two regression outputs (`reg1_pred`, `reg2_pred`).
`final_metrics_log.csv` | Aggregate `R2`, Pearson correlation, Spearman correlation, RMSE, and MAE for the testing split.



## Prediction for User Sequences

If the goal is to predict Tm values for new sequences rather than reproduce model training, run the standalone `EnzoracleTm.py` script from the `Tm/` directory:

```bash
python EnzoracleTm.py --input path/to/sequences.fasta --output prediction_results.csv
```

This script requires the three trained best checkpoints listed above. Unlike `model/test.py`, it generates ESM2 representations for new sequences during prediction and does not require a precomputed `embedding/` directory.

### Accepted Input Formats

Format | Requirements
--- | ---
FASTA (`.fasta` or `.fa`) | Use standard FASTA records: each sequence starts with a `>` identifier line followed by its amino-acid sequence.
CSV (`.csv`) | Provide a `sequence` column (or `Sequence`). An `ID` column is used when present; otherwise, the first column is used as the sequence identifier.
Text (`.txt`) | Put one amino-acid sequence on each non-empty line. Identifiers are assigned automatically as `Seq_1`, `Seq_2`, and so on.

For each input sequence, `EnzoracleTm.py` first searches `data/Tm.csv` for an exact sequence match:

- Exact matches return the database Tm value with `Source = Database Exact Match`.
- Sequences not found in the database are predicted by the integrated model with `Source = AI Model Prediction`.

The output CSV contains `ID`, `Sequence`, `Tm`, and `Source`.

Optional arguments are available when non-default paths are needed:

```bash
python EnzoracleTm.py \
  --input path/to/sequences.csv \
  --output prediction_results.csv \
  --db_path data/Tm.csv \
  --vocab_path data/data_dict.npy \
  --batch_size 8 \
  --seq_max_len 1024
```
