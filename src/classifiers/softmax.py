import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zero_like(W)

    num_train = X.shape[1]
    num_class = dW.shape[0]



    return loss, dW
