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

rng = np.random.RandomState(40)


def get_numerical_integral_benchmark(gp, t = 1.0):
    integral_compare = integrate.dblquad( lambda x,y: gp.func(tf.constant([x,y], dtype=float_type))**2, 0, t,0, t)
    return integral_compare[0]


class Test_Integration(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        #self.trainable_variables = (length_scale, unit_variance )
        
    def test_compare_integration1(self): 
        #INTEGRAL time= 1.0
        t = 1.0
        #trainable_variables = self.trainable_variables

        gp = LowRankRFF(self.length_scale , self.unit_variance, n_components = 1000, random_state = rng).fit()
        integral_out = gp.integral(lplus = t)
        integral_out = integral_out.numpy()
        
        #print(get_numerical_integral_benchmark(gp, t))
        #self.assertAlmostEqual( integral_out, 0.7442700957669496, places=7)


    def test_compare_integration2(self): 
        #INTEGRAL time= 5.0
        t = 5.0
        #trainable_variables = self.trainable_variables
         
        gp = LowRankRFF(self.length_scale , self.unit_variance, n_components = 1000, random_state = rng).fit()
        integral_out = gp.integral(lplus = t)
        integral_out = integral_out.numpy()
        
        #print(get_numerical_integral_benchmark(gp, t))
        #self.assertAlmostEqual( integral_out, 23.20454939449, places=7)
        
    def test_compare_integration3(self): 
        #INTEGRAL time= 1.0 variance = 2.0
        t = 1.0
        length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        variance =  tf.Variable(2.0, dtype=float_type, name='sig')
        
        gp = LowRankRFF(length_scale , variance, n_components = 1000, random_state = rng).fit()
        integral_out = gp.integral(lplus = t)
        integral_out = integral_out.numpy()
        
        #print(get_numerical_integral_benchmark(gp, t))
        #self.assertAlmostEqual( integral_out, 1.581768462402246, places=7)



class Test_Scaling(unittest.TestCase):
    
    def setUp(self):
        self.size = 10
        self.X = tf.constant(rng.normal(size = [self.size, 2]), dtype=float_type, name='X')
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
    
    def test_compare_integration(self): 
         
         X = self.X
         size = self.size
         unit_variance  = self.unit_variance 
         length_scale  = self.length_scale 

         gp1 = LowRankRFF(length_scale , unit_variance,n_components = 1000, random_state = rng)
         gp1.fit()

         variance = tf.Variable(2.0, dtype=float_type, name='sig')
         gp2 = LowRankRFF(length_scale , variance, n_components = 1000, random_state = rng)
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
    
    

  
    
    
