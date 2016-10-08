import unittest
import os
from classification.datautils import *


class TestDataUtils(unittest.TestCase):
    def test_load_CIFAR_batch(self):
        cifar10_dir = '/home/mininet/data/cifar-10-batches-py/data_batch_1'
        X, Y = load_CIFAR_batch(cifar10_dir)
        print 'X.shape=%s' % (X.shape,)
        print 'Y.shape=%s' % (Y.shape,)

    def test_load_CIFAR10(self):
        cifar10_dir = '/home/mininet/data/cifar-10-batches-py'
        Xtr, Ytr, Xte, Yte = load_CIFAR10(cifar10_dir)
        print 'Xtr.shape=%s' % (Xtr.shape,)
        print 'Ytr.shape=%s' % (Ytr.shape,)
        print 'Xte.shape=%s' % (Xte.shape,)
        print 'Yte.shape=%s' % (Yte.shape,)

if __name__ == '__main__':
    unittest.main()
