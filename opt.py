import numpy as np


def steepest_descent(grad, alpha=3e-4):
    move_vec = -1* alpha * grad
    return move_vec

def newton_raphson(grad, hess):
    move_vec = -1*np.dot(np.linalg.pinv(hess), grad)
    return move_vec

