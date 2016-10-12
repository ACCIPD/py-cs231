import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    W.shape = (D, C)
    X.shape = (N, D)
    y.shape = (N,)
    """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j = y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
    
    loss /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_score = 
    margins = np.zeros(scores.shape)
    margins = np.maximum(0, scores - 

    return loss, dW

