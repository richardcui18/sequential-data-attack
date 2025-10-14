# Attacks via Machine Learning: Uncovering Privacy Risks in Sequential Data Releases

## Overview
This is the official repository for Attacks via Machine Learning: Uncovering Privacy Risks in Sequential Data Releases.

## Installation
   ```bash
   pip install -r requirements.txt
   ```

## 1. Data and Preprocessing

Our analysis uses the following datasets:
- [SynMob Datasets](https://proceedings.neurips.cc/paper_files/paper/2023/file/4786c0d1b9687a841bc579b0b8b01b8e-Paper-Datasets_and_Benchmarks.pdf)
- [Geolife Dataset](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)
- [Porto Taxi Dataset](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)

Preprocessing for all datasets are done in `general_data_processing.py`.


## 2. Run Attack

You can run attack using the following command, which will first create published regions for each trajectories according to `lambda_value` and `deviation_amount_user` and then perform attacks using Bi-Directional HMM-RL algorithm and baseline (PI-uniform attack) algorithm given only the generated publish regions:

```bash
python main.py \
    --dataset <DATASET> \
    --lambda_value <LAMBDA_VALUE> \
    --deviation_amount_user <DEVIATION_AMOUNT_USER> \
    --delta <DELTA> \
    --num_iter <NUM_ITER> \
    --k <K> \
    --gamma <GAMMA>
```

- **--dataset**
Dataset to use.
- **--lambda_value**
Maximum confidence threshold determined by user (default: 0.1).
- **--deviation_amount_user**
Amount of deviation set by user when generating PR (default: 0).
- **--delta**
Threshold for reward/penalization (default: 0.3).
- **--num_iter**
Number of iterations in Bi-Directional HMM-RL algorithm (default = 100).
- **--k**
Window size in Bi-Directional HMM-RL algorithm (default = 3).
- **--gamma**
Gamma in Bi-Directional HMM-RL algorithm (default = 5).

An example command is:

```bash
python main.py \
    --dataset 'geolife' \
    --lambda_value 0.1 \
    --deviation_amount_user 1 \
    --delta 0.7 \
    --num_iter 50 \
    --k 3 \
    --gamma 5
```
