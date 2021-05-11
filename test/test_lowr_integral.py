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

import scipy.integrate as integrate
import unittest

from point.helper import get_lrgp, method
from point.point_process import Space

from gpflow.config import default_float

import arrow

rng = np.random.RandomState()


def get_numerical_integral_benchmark(gp, offset = 0):
    bounds = gp.space.bounds1D
    integral_compare = integrate.dblquad( lambda x,y: (gp.func(tf.constant([x,y], dtype=default_float())) + offset)**2, 
                                         bounds[0], bounds[1], bounds[0], bounds[1])
    return integral_compare[0]


class Test_Integration(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.variance = tf.Variable(2.0, dtype=default_float(), name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=default_float(), name='l')
        self.offset = tf.Variable(1.0, dtype=default_float(), name='o')
        self.method = method.RFF
        self.space = Space([-1,1])

    def test_compare_integration(self): 
        
        lrgp = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
    
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_benchmark(lrgp)
        
        print("[{}] TEST Integration:".format(arrow.now()))
        print("[{}] integral.calculation:= : {}".format(arrow.now(), integral_out))
        print("[{}] numerical.integration:= {}".format(arrow.now(), integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
    
    def test_compare_integration_with_offset(self): 
        
        lrgp = get_lrgp(method = self.method, variance = self.variance, beta0 = self.offset, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_benchmark(lrgp, self.offset.numpy())
        
        print("[{}] TEST Integration.with.offset:".format(arrow.now()))
        print("[{}] integral.calculation:= : {}".format(arrow.now(), integral_out))
        print("[{}] numerical.integration:= {}".format(arrow.now(), integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        



      
if __name__ == '__main__':
    unittest.main()
    
