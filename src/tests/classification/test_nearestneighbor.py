import numpy as np
import unittest
import classification

from classification.nearestneighbor import NearestNeighbor


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

if __name__ == '__main__':
    unittest.main()
