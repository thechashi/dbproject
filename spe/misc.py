import numpy as np


def get_filled_matrix(n_row, n_column, value):
    """


    Parameters
    ----------
    n_row : TYPE
        DESCRIPTION.
    n_column : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    matrix : TYPE
        DESCRIPTION.

    """
    matrix = np.full(shape=(n_row, n_column), fill_value=value)
    return matrix


def random_sampling(major_X, major_y, minor_X, minor_y, random_seed_value):
    """


    Parameters
    ----------
    major_X : TYPE
        DESCRIPTION.
    major_y : TYPE
        DESCRIPTION.
    minor_X : TYPE
        DESCRIPTION.
    minor_y : TYPE
        DESCRIPTION.
    random_seed_value : TYPE
        DESCRIPTION.

    Returns
    -------
    train_X : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.

    """

    np.random.seed(random_seed_value)
    limit = len(major_X)
    total = len(minor_X)
    idx = np.random.choice(
        limit, total, replace=False
    )  # getting similar number of sample indexes like minority set

    train_X = np.concatenate(
        [major_X[idx], minor_X]
    )  # new train X with balanced majority and minorty samples
    train_y = np.concatenate(
        [major_y[idx], minor_y]
    )  # new train y with balanced majority and minorty samples

    return train_X, train_y


def process_array(array):
    """


    Parameters
    ----------
    array : TYPE
        DESCRIPTION.

    Returns
    -------
    array : TYPE
        DESCRIPTION.

    """
    array[np.isnan(array)] = 0
    return array


def stack_majority_minority(majority_X, minority_X, majority_y, minority_y):
    """


    Parameters
    ----------
    majority_X : TYPE
        DESCRIPTION.
    minority_X : TYPE
        DESCRIPTION.
    majority_y : TYPE
        DESCRIPTION.
    minority_y : TYPE
        DESCRIPTION.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    X = np.vstack([majority_X, minority_X])
    y = np.hstack([majority_y, minority_y])
    return X, y


def get_maximum_indexes(array, max_value):
    """


    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    max_value : TYPE
        DESCRIPTION.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    result = array == max_value
    return result
