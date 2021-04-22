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

from point.low_rank_gp import LowRankApproxGP

rng = np.random.RandomState(40)



class Test_Integration(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        

    def test_compare_integration1(self): 
        t = 1.0
        length_scale = self.length_scale
        unit_variance = self.unit_variance

        gp = LowRankApproxGP(n_components = 100, random_state = rng)
        gp.fit(length_scale, unit_variance)
        integral_out, _ = gp.sq_integral_grad(length_scale, unit_variance)
        integral_out = integral_out.numpy()
        
        integral_compare = integrate.dblquad( lambda x,y: gp.func(tf.constant([x,y], dtype=float_type))**2, 0, t,0, t)
        self.assertAlmostEqual( integral_out, integral_compare[0], places=7)

        
        
    def test_compare_integration2(self): 
        t = 5.0
        length_scale = self.length_scale
        unit_variance = self.unit_variance
        
        gp = LowRankApproxGP(n_components = 100, random_state = rng)
        gp.fit(length_scale, unit_variance)
        integral_out, _ = gp.sq_integral_grad(length_scale, unit_variance, T = t)
        integral_out = integral_out.numpy()
        
        integral_compare = integrate.dblquad( lambda x,y: gp.func(tf.constant([x,y], dtype=float_type))**2, 0, t,0, t)
        self.assertAlmostEqual( integral_out, integral_compare[0], places=7)



class Test_Scaling(unittest.TestCase):
    
    def setUp(self):
        self.size = 10
        self.X = tf.constant(rng.normal(size = [self.size, 2]), dtype=float_type, name='X')
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
    
    def test_compare_integration1(self): 
         
         X = self.X
         size = self.size
         
         length_scale = self.length_scale
         unit_variance = self.unit_variance
         
         gp1 = LowRankApproxGP(n_components = 1000, random_state = rng)
         gp1.fit(length_scale, unit_variance)

         variance = tf.Variable(2.0, dtype=float_type, name='sig')
         gp2 = LowRankApproxGP(n_components = 1000, random_state = rng)
         gp2.fit(length_scale, variance)
        
         gp2.randomFourier_.random_weights_ = gp1.randomFourier_.random_weights_
         gp2.randomFourier_.random_offset_ = gp1.randomFourier_.random_offset_
         gp2.beta_ = gp1.beta_
         
         f1 = gp1.func(X).numpy()
         f2 = gp2.func(X).numpy()
         
         scaled_f1 = f1 * np.sqrt(variance.numpy())
         
         for i in range(size):
             self.assertAlmostEqual( scaled_f1[i][0], f2[i][0], places=7)


      
if __name__ == '__main__':
    unittest.main()
    
    

  
    
    
