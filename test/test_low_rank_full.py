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

float_type = tf.dtypes.float64

import scipy.integrate as integrate
import unittest
import copy

from point.low_rank_rff import LowRankRFF
from point.helper import get_lrgp, method
from point.point_process import Space

rng = np.random.RandomState()


def get_numerical_integral_benchmark(gp):
    bounds = gp.space.bounds1D
    integral_compare = integrate.dblquad( lambda x,y: gp.func(tf.constant([x,y], dtype=float_type))**2, bounds[0], bounds[1],bounds[0], bounds[1])
    return integral_compare[0]


class Test_Integration(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.unit_variance = tf.Variable(2.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        self.method = method.RFF
        self.space = Space([-1,1])
        
        self.lrgp = get_lrgp(method = self.method, length_scale = self.length_scale, space = self.space,
                             variance = self.unit_variance, n_components = 250, random_state = rng).fit()

    def test_compare_integration(self): 
        #INTEGRAL time= 1.0
        integral_out = self.lrgp.integral()
        integral_out = integral_out.numpy()
        
        integral_recomputed = get_numerical_integral_benchmark(self.lrgp)
        
        print(integral_out)
        print(integral_recomputed)
        #self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        


class Test_Scaling(unittest.TestCase):
    
    def setUp(self):
        self.size = 10
        self.space = Space([0,1])
        self.X = tf.constant(rng.normal(size = [self.size, 2]), dtype=float_type, name='X')
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
    
    def test_compare(self): 
         
         X = self.X
         size = self.size
         space = Space([0,1])
         unit_variance  = self.unit_variance 
         length_scale  = self.length_scale 

         gp1 = LowRankRFF(length_scale , unit_variance, space = space, n_components = 1000, random_state = rng)
         gp1.fit()

         variance = tf.Variable(2.0, dtype=float_type, name='sig')
         gp2 = LowRankRFF(length_scale , variance, space = space, n_components = 1000, random_state = rng)
         gp2.fit()
        
         gp2.random_weights_ = gp1.random_weights_
         gp2.random_offset_ = gp1.random_offset_
         gp2.beta_ = gp1.beta_
         
         f1 = gp1.func(X).numpy()
         f2 = gp2.func(X).numpy()
         
         scaled_f1 = f1 * np.sqrt(variance.numpy())
         
         for i in range(size):
             self.assertAlmostEqual( scaled_f1[i][0], f2[i][0], places=7)


      
if __name__ == '__main__':
    unittest.main()
    
    

  
    
    
