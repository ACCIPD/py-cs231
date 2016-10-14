import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    W(D, C)
    X(N, D)
    Y(N,)
    """

    loss = 0.0
    dW = np.zero_like(W)

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in xrange(num_train):
        scores_i = X[i].dot(W)
        scores_i -= np.max(scores)

        loss_i = -scores_i[y[i]] + np.log(np.sum(np.exp(scores_i)))
        loss += loss_i

    loss = loss / float(num_train) + 0.5 * reg * np.sum(W * W)        

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zero(W.shape)

    scores = X.dot(W)
    scores -= np.max(scores, axis = 1).reshape(-1, 1)
    loss = -scores[np.arange(num_train), y] + np.log(np.sum(np.exp(scores), axis = 1))

    loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W * W)

    return loss, dW

