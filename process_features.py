import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.utils import shuffle
import torch
import random
import os


def main():
    save_dir = "./pca_data"
    os.makedirs(save_dir, exist_ok=True)

    data1 = pd.read_csv('features/MHC_ESM1b_1.csv')
    data2 = pd.read_csv('features/MHC_ESM2_1.csv')
    data3 = pd.read_csv('features/MHC_ESM1b_2.csv')
    data4 = pd.read_csv('features/MHC_ESM2_2.csv')

    labels = data1.iloc[:, 0].values
    f1 = data1.iloc[:, 1:].values
    f2 = data2.iloc[:, 1:].values
    f3 = data3.iloc[:, 1:].values
    f4 = data4.iloc[:, 1:].values


    X = np.concatenate([f1, f3, f2, f4], axis=1)


    X, y = shuffle(X, labels, random_state=42)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=42
    )


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=224, random_state=0)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    np.save(f"{save_dir}/best_train_X.npy", X_train_pca)
    np.save(f"{save_dir}/best_train_y.npy", y_train)
    np.save(f"{save_dir}/best_train_idx.npy", train_idx)

    np.save(f"{save_dir}/best_test_X.npy", X_test_pca)
    np.save(f"{save_dir}/best_test_y.npy", y_test)
    np.save(f"{save_dir}/best_test_idx.npy", test_idx)

    model = MLPClassifier(random_state=0)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    acc = round(accuracy_score(y_test, y_pred), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)

    print(f"Test Accuracy: {acc}")

if __name__ == "__main__":
    main()
