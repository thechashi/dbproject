from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def load_sklearn_data(test_size=0.1, minority_label=7, random_state=41):
    """


    Parameters
    ----------
    test_size : TYPE, optional
        DESCRIPTION. The default is 0.1.
    minority_label : TYPE, optional
        DESCRIPTION. The default is 7.
    random_state : TYPE, optional
        DESCRIPTION. The default is 41.

    Returns
    -------
    train_X : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    test_X : TYPE
        DESCRIPTION.
    test_y : TYPE
        DESCRIPTION.

    """

    print("Loading forest imbalanced dataset... ")
    X, y = datasets.fetch_covtype(return_X_y=True)

    # turning into binary classes where label 7 (minority label) will be 1 and
    # labels 1,2,3,4,5, and 6 will be 0
    indexes_of_seven = y == minority_label
    y[indexes_of_seven] = 1  # minority labels
    y[~indexes_of_seven] = 0  # majority labels

    majority_count = (y == 0).sum()
    minority_count = (y == 1).sum()
    imbalance_ratio = majority_count / minority_count

    majority_X = X[y == 0]
    majority_y = y[y == 0]
    minority_X = X[y == 1]
    minority_y = y[y == 1]

    (
        train_majority_X,
        test_majority_X,
        train_majority_y,
        test_majority_y,
    ) = train_test_split(
        majority_X, majority_y, test_size=test_size, random_state=random_state
    )

    (
        train_minority_X,
        test_minority_X,
        train_minority_y,
        test_minority_y,
    ) = train_test_split(
        minority_X, minority_y, test_size=test_size, random_state=random_state
    )
    train_X = np.concatenate([train_majority_X, train_minority_X])
    train_y = np.concatenate([train_majority_y, train_minority_y])

    test_X = np.concatenate([test_majority_X, test_minority_X])
    test_y = np.concatenate([test_majority_y, test_minority_y])

    print("\nData loaded.")
    print("\nMajority count: {}".format(majority_count))
    print("Minority count: {}".format(minority_count))
    print("Imbalance Ratio: {}".format(imbalance_ratio))
    print()
    return train_X, train_y, test_X, test_y


def split_dataset(X, y, test_size, random_state=10):
    """


    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    test_size : TYPE
        DESCRIPTION.
    random_state : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    train_X : TYPE
        DESCRIPTION.
    test_X : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    test_y : TYPE
        DESCRIPTION.

    """
    majority_X = X[y == 0]
    majority_y = y[y == 0]
    minority_X = X[y == 1]
    minority_y = y[y == 1]
    (
        train_majority_X,
        test_majority_X,
        train_majority_y,
        test_majority_y,
    ) = train_test_split(
        majority_X, majority_y, test_size=test_size, random_state=random_state
    )

    (
        train_minority_X,
        test_minority_X,
        train_minority_y,
        test_minority_y,
    ) = train_test_split(
        minority_X, minority_y, test_size=test_size, random_state=random_state
    )
    train_X = np.concatenate([train_majority_X, train_minority_X])
    train_y = np.concatenate([train_majority_y, train_minority_y])

    test_X = np.concatenate([test_majority_X, test_minority_X])
    test_y = np.concatenate([test_majority_y, test_minority_y])

    return train_X, test_X, train_y, test_y


def load_custom_data(filepath, test_size=0.1, total_samples=6000, random_state=10):
    """


    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    test_size : TYPE, optional
        DESCRIPTION. The default is 0.1.
    total_samples : TYPE, optional
        DESCRIPTION. The default is 6000.
    random_state : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    train_X : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    test_X : TYPE
        DESCRIPTION.
    test_y : TYPE
        DESCRIPTION.

    """
    dataset = pd.read_csv(filepath)

    print("Loading custom imbalanced dataset... ")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    majority_count = (y == 0).sum()
    minority_count = (y == 1).sum()
    imbalance_ratio = majority_count / minority_count

    majority_X = X[y == 0]
    majority_y = y[y == 0]
    minority_X = X[y == 1]
    minority_y = y[y == 1]

    (
        train_majority_X,
        test_majority_X,
        train_majority_y,
        test_majority_y,
    ) = train_test_split(
        majority_X, majority_y, test_size=test_size, random_state=random_state
    )

    (
        train_minority_X,
        test_minority_X,
        train_minority_y,
        test_minority_y,
    ) = train_test_split(
        minority_X, minority_y, test_size=test_size, random_state=random_state
    )

    train_X = np.concatenate([train_majority_X, train_minority_X])
    train_y = np.concatenate([train_majority_y, train_minority_y])

    test_X = np.concatenate([test_majority_X, test_minority_X])
    test_y = np.concatenate([test_majority_y, test_minority_y])

    print("\nData loaded.")
    print("\nMajority count: {}".format(majority_count))
    print("Minority count: {}".format(minority_count))
    print("Imbalance Ratio: {}".format(imbalance_ratio))
    print()
    return train_X, train_y, test_X, test_y


def load_finance_data(filepath, test_size=0.1, random_state=10):
    """


    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    test_size : TYPE, optional
        DESCRIPTION. The default is 0.1.
    random_state : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    train_X : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    test_X : TYPE
        DESCRIPTION.
    test_y : TYPE
        DESCRIPTION.

    """
    dataset = pd.read_csv(filepath)

    print("Loading custom imbalanced dataset... ")
    dt = dataset.iloc[:, [0, 2, 4, 5, 7, 8, 9, 10]].values

    X = dt[:, :-1]
    y = dt[:, -1]

    majority_count = (y == 0).sum()
    minority_count = (y == 1).sum()
    imbalance_ratio = majority_count / minority_count

    majority_X = X[y == 0]
    majority_y = y[y == 0]
    minority_X = X[y == 1]
    minority_y = y[y == 1]

    (
        train_majority_X,
        test_majority_X,
        train_majority_y,
        test_majority_y,
    ) = train_test_split(
        majority_X, majority_y, test_size=test_size, random_state=random_state
    )

    (
        train_minority_X,
        test_minority_X,
        train_minority_y,
        test_minority_y,
    ) = train_test_split(
        minority_X, minority_y, test_size=test_size, random_state=random_state
    )

    train_X = np.concatenate([train_majority_X, train_minority_X])
    train_y = np.concatenate([train_majority_y, train_minority_y])

    test_X = np.concatenate([test_majority_X, test_minority_X])
    test_y = np.concatenate([test_majority_y, test_minority_y])

    print("\nData loaded.")
    print("\nMajority count: {}".format(majority_count))
    print("Minority count: {}".format(minority_count))
    print("Imbalance Ratio: {}".format(imbalance_ratio))
    print()
    return train_X, train_y, test_X, test_y
