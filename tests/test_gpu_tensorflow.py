import unittest
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_allclose, assert_, assert_warns)


import tensorflow as tf
import numpy as np
import scipy as sp


class TestTensorFlowMethods(unittest.TestCase):
    def test_gpu_exists(self):
        print("Tensflow version: ", tf.__version__)
        physical_devices = tf.config.list_physical_devices('GPU')
        self.assertGreater(len(physical_devices), 0, "No GPU Found")


if __name__ == '__main__':
    unittest.main()
