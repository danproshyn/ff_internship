import re

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


def split_dataset(clients_path, features_path, valid_size=0.2):
    clients = pd.read_csv(clients_path)
    features = pd.read_csv(features_path)

    dataset = pd.merge(clients, features, on='client_id')
    dataset = dataset.drop(columns=['client_id', 'communication_month'])

    train_dataset = dataset[dataset['is_train']].drop(columns=['is_train'])
    test_dataset = dataset[~dataset['is_train']].drop(columns=['is_train'])

    X = train_dataset.drop(columns=['target'])
    X_test = test_dataset.drop(columns=['target'])
    y = train_dataset['target']
    y_test = test_dataset['target']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, stratify=y, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_dataset_v2(path, valid_size=0.2, replace_special_symbols=False):
    dataset = pd.read_csv(path, low_memory=False, dtype={'target': 'bool', 'is_train': 'boolean'})
    dataset.drop(columns=['client_id'], inplace=True)

    if replace_special_symbols:
        dataset = dataset.rename(columns=lambda x: re.sub(r'[,\n\[\]\{\}:"]', '__', x))

    train_dataset = dataset[dataset['is_train'] == 1].drop(columns=['is_train'])
    test_dataset = dataset[dataset['is_train'] == 0].drop(columns=['is_train'])

    X = train_dataset.drop(columns=['target'])
    X_test = test_dataset.drop(columns=['target'])
    y = train_dataset['target']
    y_test = test_dataset['target']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, stratify=y, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def calc_gini_coef(y_true, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * roc_auc - 1


def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.plot(fpr, tpr, label='auc=' + str(roc_auc))
    plt.legend(loc='lower right')
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def plot_pr_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall, precision)
    plt.title('PR-curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
