import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    matthews_corrcoef, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
import numpy as np
import random
import torch
import pickle

from tqdm import tqdm


manualSeed = 2
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data1 = pd.read_csv('features/MHC_ESM.csv')  # shape: [N, 1280]
    data1_new = np.load('features/MHC_features.npy')
    data2_new = np.load('features/MHC_ESM2_features.npy')
    data2 = pd.read_csv('features/MHC_ESM2-650M.csv')
    data3 = pd.read_csv('features/MHC_ESM1b-650M.csv')
    data4 = pd.read_csv('features/MHC_ESM2.csv')
    data5 = pd.read_csv('features/MHC_2_windows.csv')


    data_manual1 = pd.read_csv('features/manul_process.csv')

    labels = data1.iloc[:, 0].values
    f1 = data1.iloc[:, 1:].values
    f1_new = data1_new
    f2 = data2.iloc[:, 1:].values
    f2_new = data2_new
    f3 = data3.iloc[:, 1:].values
    f4 = data4.iloc[:, 1:].values

    f5 = data5

    m_ACC = data_manual1.iloc[:, 1:].values

    X = np.concatenate([f1,f2,f3,f4], axis=1)

    # 打乱数据集
    features_shuffled, labels_shuffled = shuffle(X, labels, random_state=42)
    # 数据集划分
    train_features, test_features, train_labels, test_labels = train_test_split(features_shuffled, labels_shuffled,
                                                                                test_size=0.2, random_state=42)
    # 标准化特征
    '''
    创建一个标准化器的实例。
    标准化的目的是将特征数据转换为均值为0，标准差为1的分布。
    这有助于消除特征之间的量纲差异，使得模型训练更稳定，更容易收敛。
    '''
    scaler = StandardScaler()
    origin_train_features = scaler.fit_transform(train_features)
    origin_test_features = scaler.transform(test_features)

    best_test_acc = 0
    best_test_sn = 0
    best_test_sp = 0
    best_test_mcc = 0

    best_train_acc = 0
    best_train_sn = 0
    best_train_sp = 0
    best_train_mcc = 0

    best_train_feature = 0
    best_test_feature = 0

    train_acc_records = []  # 用于记录 (特征数量, train_acc)
    for i in tqdm(range(1, 400)):

        print(f"feature: {i}")
        pca = PCA(n_components=i, random_state=0)
        train_features = pca.fit_transform(origin_train_features)

        # kernel_pca

        # train_features = pca.fit_transform(origin_train_features)

        model = MLPClassifier()

        cv = StratifiedKFold(n_splits=10)
        scoring = {
            'acc': make_scorer(accuracy_score),
            'sp': make_scorer(precision_score, average='macro'),
            'sn': make_scorer(recall_score, average='macro'),
            'mcc': make_scorer(matthews_corrcoef),
            'auc': make_scorer(roc_auc_score, needs_proba=True)
        }

        # 十折交叉验证结果
        results = cross_validate(model, origin_train_features, train_labels, cv=cv, scoring=scoring)
        acc_cv = round(results['test_acc'].mean(), 4)
        sp_cv = round(results['test_sp'].mean(), 4)
        sn_cv = round(results['test_sn'].mean(), 4)
        mcc_cv = round(results['test_mcc'].mean(), 4)
        auc_cv = round(results['test_auc'].mean(), 4)
        print(f"Train: acc_cv: {acc_cv}, sp_cv: {sp_cv}, sn_cv: {sn_cv}, mcc_cv: {mcc_cv}")

        # 保存当前特征数量与acc
        train_acc_records.append((i, acc_cv))

        if acc_cv > best_train_acc:
            best_train_acc = acc_cv
            best_train_sn = sn_cv
            best_train_sp = sp_cv
            best_train_mcc = mcc_cv
            best_train_feature = i

        # 训练
        model.fit(train_features, train_labels)
        # 封装模型
        # 'pca': pca,
        model_bundle = {
            'scaler': scaler,
            'pca': pca,
            'model': model
        }


        # 测试集结果
        with open('model_bundle_MLP.pkl', 'wb') as file:
            pickle.dump(model_bundle, file)

            scaler = model_bundle['scaler']
            pca = model_bundle['pca']
            model = model_bundle['model']


            # test_features = pca.transform(origin_test_features)

            test_features = scaler.transform(test_features)
            test_features = pca.transform(test_features)

            predictions = model.predict(test_features)

            acc_test = round(accuracy_score(test_labels, predictions), 4)
            sp_test = round(precision_score(test_labels, predictions, average='macro'), 4)
            sn_test = round(recall_score(test_labels, predictions, average='macro'), 4)
            mcc_test = round(matthews_corrcoef(test_labels, predictions), 4)


            print(f"Test: acc: {acc_test}, sp: {sp_test}, sn: {sn_test}, mcc: {mcc_test}")
            if acc_cv > best_test_acc:
                best_test_acc = acc_cv
                best_test_sn = sn_cv
                best_test_sp = sp_cv
                best_test_mcc = mcc_cv
                best_test_feature = i



    print(f"train feature{best_train_feature} acc:{best_train_acc}, sp:{best_train_sp}, sn:{best_train_sn}, mcc:{best_train_mcc}")
    print(
        f"test feature{best_test_feature} acc:{best_test_acc}, sp:{best_test_sp}, sn:{best_test_sn}, mcc:{best_test_mcc}")


if __name__ == '__main__':
    main()