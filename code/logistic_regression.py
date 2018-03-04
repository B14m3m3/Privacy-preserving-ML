import numpy as np
import util

def logistic(z):
    """
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z
    Args:
        z: numpy array shape (d,)
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function
    """
    logi = 1/(1+np.exp(-z))

    assert logi.shape == z.shape
    return logi

def log_cost(X, y, w, C, order, reg=0):
    """
    Args:
        X: np.array shape (n,d) float - Features
        y: np.array shape (n,)  int - Labels
        w: np.array shape (d,)  float - Initial parameter vector
        reg: scalar - regularization parameter

    Returns:
      cost: scalar the cross entropy cost of logistic regression with data X,y using regularization parameter reg
      grad: np.arrray shape(n,d) gradient of cost at w with regularization value reg
    """
    gradients = None
    grad_reg = reg * w
    grad_reg[0] = 0
    for i, data in enumerate(X):
        grad = -(data * (y[i]-logistic(np.dot(data, w))))
        gradients = util.clip(C, grad, order) if gradients is None else gradients + util.clip(C, grad, order)
    gradients = gradients + grad_reg
    assert gradients.shape == w.shape
    return gradients
