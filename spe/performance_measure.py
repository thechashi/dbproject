from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    matthews_corrcoef,
)
import numpy as np


def accuracy(y_true, y_hat):
    """


    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_hat : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    result = accuracy_score(y_true, y_hat)
    return result


def precision_and_recall(label, y_pred):
    """


    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    precision : TYPE
        DESCRIPTION.
    recall : TYPE
        DESCRIPTION.

    """
    temp = y_pred.copy()
    precision, recall, _ = precision_recall_curve(label, temp)

    return precision, recall


def aucprc(label, y_pred):
    """


    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    result = average_precision_score(label, y_pred)
    return result


def f1_score_optimal(label, y_pred):
    """


    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    precision, recall = precision_and_recall(label, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    result = max(f1_scores)
    return result


def g_mean_score_optimal(label, y_pred):
    """


    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    precision, recall = precision_and_recall(label, y_pred)
    gms = np.power((precision * recall), (1 / 2))
    result = max(gms)
    return result


def mat_corrcoef(label, y_pred):
    """


    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    temp = y_pred.copy()
    result = matthews_corrcoef(label, temp)
    return result


def mcc_score_optimal(label, y_pred):
    """


    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    mcc_list = []
    for t in range(100):
        temp = y_pred.copy()
        temp[temp < 0 + t * 0.01] = 0
        temp[temp >= 0 + t * 0.01] = 1
        mcc = mat_corrcoef(label, temp)
        mcc_list.append(mcc)
    result = max(mcc_list)
    return result
