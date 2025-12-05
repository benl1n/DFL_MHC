# DFL-MHC

## 1. Introduction

DFL-MHC is a dual-stage learning framework for MHC protein identification that integrates multi-view feature fusion from multiple protein language models with BiLSTM-based deep sequence modeling.

![Model Architecture](https://github.com/benl1n/DFL-MHC/blob/main/DFL-MHC.png)

## 2. Python Environment

Python 3.10 and packages version:

- torch == 2.5.1
- numpy ==1.26.3

- tqdm == 4.67.0
- scikit-learn == 1.5.1

## 3.Prediction

1. Using Git:

```
git clone https://github.com/benl1n/DFL-MHC.git
```

download the pre-computed features produced by the ESM models from this shared folder: https://drive.google.com/drive/folders/13aU8T6qJt2xp3VmJswOXmakMGHZBl6zf?usp=drive_link

create a local directory named `features/` and save the following CSV files there:

- MHC_ESM1b_1.csv

- MHC_ESM1b_2.csv

- MHC_ESM2_1.csv
- MHC_ESM2_2.csv

2. search for the optimal feature dimension using PCA and MLP within the range of 1 to 400 :

```python
python first_stage.py
```

3. save the optimal features as `.npy` for final training and testing:

```
python process_features.py
```

4. get the final prediction results of MHC:

```
python train_test.py
```













