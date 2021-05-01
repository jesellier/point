# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf

import gpflow.kernels as gfk

float_type = tf.dtypes.float64

rng = np.random.RandomState(40)

import unittest


class Test_Kernel(unittest.TestCase):
    

    def setUp(self):
        rng  = np.random.RandomState(8)
        variance = tf.Variable(1, dtype=float_type, name='sig')
        length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='lenght_scale')

        self.X = tf.constant(rng.normal(size = [500, 2]), dtype=float_type, name='X')
        self.kernel = gfk.SquaredExponential(variance= variance , lengthscales= length_scale)


    def test_compare(self):
        pass




if __name__ == '__main__':
    unittest.main()

 
    
    