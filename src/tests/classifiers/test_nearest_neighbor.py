import numpy as np
import unittest
import classification

from classification.nearestneighbor import NearestNeighbor
from classification.datautils import load_CIFAR10

class TestNearestNeighbor(unittest.TestCase):
    def test_predict(self):
        nn = NearestNeighbor()
        Xtr = np.arange(6).reshape(2,3)
        print "Xtr=" 
        print Xtr
        Ytr = np.arange(2)
        print "Ytr=" 
        print Ytr
        nn.train(Xtr, Ytr)
        Ypred = nn.predict(Xtr)

        print Ypred

    def test_nn_cifar10(self):
        Xtr, Ytr, Xte, Yte = load_CIFAR10('/home/mininet/data/cifar-10-batches-py')
        Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
        Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

        nn = NearestNeighbor()
        nn.train(Xtr_rows, Ytr)
        Yte_predict = nn.predict(Xte_rows)

        print 'accuracy: %f' % (np.mean(Yte_predict == Yte))


if __name__ == '__main__':
    unittest.main()
