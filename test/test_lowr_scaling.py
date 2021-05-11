# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:07:51 2021

@author: jesel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import unittest

from point.helper import get_lrgp, method
from point.point_process import Space

from gpflow.config import default_float

rng = np.random.RandomState()


class Test_Scaling(unittest.TestCase):
    
    def setUp(self):
        self.size = 10
        self.space = Space([0,1])
        self.X = tf.constant(rng.normal(size = [self.size, 2]), dtype=default_float(), name='X')
        self.method = method.RFF
        
        self.unit_variance = tf.Variable(1.0, dtype=default_float(), name='sig')
        self.variance = tf.Variable(2.0, dtype=default_float(), name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=default_float(), name='l')
    
    def test_compare(self): 

         gp1 = get_lrgp(method = self.method, variance = self.unit_variance, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        
         gp2 = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        
         gp2._random_weights = gp1._random_weights
         gp2._random_offset = gp1._random_offset
         gp2._latent = gp1._latent
        
         f1 = gp1.func(self.X).numpy()
         f2 = gp2.func(self.X).numpy()

         scaled_f1 = f1 * np.sqrt(self.variance.numpy())
        
         for i in range(self.size):
            self.assertAlmostEqual( scaled_f1[i][0], f2[i][0], places=7)


      
if __name__ == '__main__':
    unittest.main()
    
    

 