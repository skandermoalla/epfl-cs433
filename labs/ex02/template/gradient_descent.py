# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""

from ex02.template.costs import compute_loss


def compute_gradient(y, tx, w, loss_name="MSE"):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    if loss_name == "MSE":
        return compute_gradient_mse(y, tx, w)
    elif loss_name == "MAE":
        return compute_gradient_mae(y, tx, w)
    else:
        raise NotImplementedError


def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return - tx.T.dot(e) / y.shape[0]


def compute_gradient_mae(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return - tx.T.dot(e) / y.shape[0]


def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_name="MSE"):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
            [loss(w0), loss(w1), ..., loss(w_max_iters-1)]
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
            [w0, w1, ..., w_max_iters-1]
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_name=loss_name)
        grad_w = compute_gradient(y, tx, w, loss_name=loss_name)
        w -= gamma * grad_w
        ws.append(w)
        losses.append(loss)

        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters-1, l=loss, w0=ws[-2][0], w1=ws[-2][1]))

    return losses, ws[:-1]
