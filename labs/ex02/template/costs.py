# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w, loss_name="MSE"):
    """Calculate the loss using the MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # needs to be redefined depending on the need.
    # This is because of the poor design of the tutorials.
    if loss_name == "MSE":
        return compute_mse(y, tx, w)
    elif loss_name == "MAE":
        return compute_mae(y, tx, w)
    else:
        raise NotImplementedError


def compute_mse(y, tx, w):
    """Calculate the loss using the MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    return (e ** 2).mean() / 2


def compute_mae(y, tx, w):
    """Calculate the loss using the MAE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    return e.abs().mean()
