# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # ridge regression: TODO
    # ***************************************************
    A = tx.T @ tx + lambda_ * 2 * tx.shape[0] * np.diag(np.ones(tx.shape[1]))
    b = tx.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.LinAlgError:
        print("using lstsq")
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

