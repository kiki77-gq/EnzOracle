# **EnzOracle: Distribution-aware prediction of enzyme environmental adaptation via a classification-guided mixture-of-experts framework**

![Graphical abstract](https://raw.githubusercontent.com/kiki77-gq/EnzOracle/main/Graphical%20Abstract%20Image.png)

EnzOracle is a classification-guided mixture-of-experts framework for predicting enzyme environmental-adaptation properties from protein sequences.  Each task first uses a classifier to identify the densely sampled target interval, then combines the corresponding interval-specific regression expert with a complementary regression expert for the remaining target range.  This distribution-aware design is intended to improve predictions where the experimental-data distribution is uneven.

## Branches

The repository contains three task-specific implementation branches.  Each branch has its own complete README with data, training, evaluation, and prediction details.

| Branch | Predicted property | Classification-guided expert design | Detailed documentation |
| --- | --- | --- | --- |
| [`Tm`](https://github.com/kiki77-gq/EnzOracle/tree/Tm) | Melting temperature (`Tm`)| [`Tm` README](https://github.com/kiki77-gq/EnzOracle/blob/Tm/README.md) |
| [`Topt`](https://github.com/kiki77-gq/EnzOracle/tree/Topt) | Optimum temperature (`Topt`) | [`Topt` README](https://github.com/kiki77-gq/EnzOracle/blob/Topt/README.md) |
| [`pHopt`](https://github.com/kiki77-gq/EnzOracle/tree/pHopt) | Optimum pH (`pHopt`) | [`pHopt` README](https://github.com/kiki77-gq/EnzOracle/blob/pHopt/README.md) |

## Minimal environment

The code was configured and checked against the existing `EnzOracle` environment.  Use an NVIDIA GPU setup compatible with CUDA 11.3 and the following core software versions:

| Component | Version |
| --- | --- |
| Python | 3.9.23 |
| PyTorch | 1.12.1 |
| CUDA | 11.3 |
| fair-esm | 2.0.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |


The three branches use ESM2 protein-language-model representations (`esm2_t33_650M_UR50D`).  Please follow the branch-specific README for the task data, embedding preparation, checkpoints, and execution procedures.

## Repository

Source code and branch-specific documentation: [kiki77-gq/EnzOracle](https://github.com/kiki77-gq/EnzOracle)
