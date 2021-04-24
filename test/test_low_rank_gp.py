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

from point.low_rank_gp import LowRankApproxGP

rng = np.random.RandomState(40)


def get_numerical_integral_benchmark(gp, t = 1.0):
    integral_compare = integrate.dblquad( lambda x,y: gp.func(tf.constant([x,y], dtype=float_type))**2, 0, t,0, t)
    return integral_compare[0]


class Test_access(unittest.TestCase):
    
    def setUp(self):
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        self.gp = LowRankApproxGP(n_components = 100, random_state = rng).fit(self.length_scale, self.unit_variance)
        self.beta_ = copy.deepcopy(self.gp.beta_)
        
    def test_get(self):
        tmp = self.gp.trainable_variables()
        self.assertTrue(self.unit_variance == tmp['variance'])
        self.assertTrue(((self.length_scale).numpy() == (tmp['length_scale']).numpy()).all())
        
    def test_set(self):
        v2 = tf.Variable(10.0, dtype=float_type, name='sig')
        l2 = tf.Variable([0.5,0.5], dtype=float_type, name='l')

        #test correct passing of the new arguments
        self.gp.reset_trainable_variables(v2, l2)
        tmp = self.gp.trainable_variables()
        self.assertTrue(tmp['variance'] == v2)
        self.assertTrue(((tmp['length_scale']).numpy() == l2.numpy()).all())

        #check other variable are left unchanged
        self.assertTrue((self.beta_.numpy() == self.gp.beta_.numpy()).all())
        
        #test the parameters change has been passed to the nested randomFourrier object
        gamma = 1 / (2 * l2 **2 )
        tmp = self.gp.randomFourier_.trainable_variables()
        self.assertTrue(tmp['variance'] == v2)
        self.assertTrue(((tmp['gamma']).numpy() == gamma.numpy()).all())



class Test_Integration(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        
    def test_compare_integration1(self): 
        #INTEGRAL time= 1.0
        t = 1.0
        length_scale = self.length_scale
        unit_variance = self.unit_variance

        gp = LowRankApproxGP(n_components = 100, random_state = rng).fit(length_scale, unit_variance)
        integral_out, _ = gp.integral_grad(length_scale, unit_variance, T = t)
        integral_out = integral_out.numpy()
        
        #print(get_numerical_integral_benchmark(gp, t))
        self.assertAlmostEqual( integral_out, 1.431831880626388, places=7)


    def test_compare_integration2(self): 
        #INTEGRAL time= 5.0
        t = 5.0
        length_scale = self.length_scale
        unit_variance = self.unit_variance
        
        gp = LowRankApproxGP(n_components = 100, random_state = rng).fit(length_scale, unit_variance)
        integral_out, _ = gp.integral_grad(length_scale, unit_variance, T = t)
        integral_out = integral_out.numpy()
        
        #print(get_numerical_integral_benchmark(gp, t))
        self.assertAlmostEqual( integral_out, 17.689008254722765, places=7)
        
    def test_compare_integration3(self): 
        #INTEGRAL time= 1.0 variance = 2.0
        t = 1.0
        length_scale = self.length_scale
        unit_variance =  tf.Variable(2.0, dtype=float_type, name='sig')
        
        gp = LowRankApproxGP(n_components = 100, random_state = rng).fit(length_scale, unit_variance)
        integral_out, _ = gp.integral_grad(length_scale, unit_variance, T = t)
        integral_out = integral_out.numpy()
        
        #print(get_numerical_integral_benchmark(gp, t))
        self.assertAlmostEqual( integral_out, 2.311289874624141, places=7)



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
         gp1.fit(length_scale , unit_variance, )

         variance = tf.Variable(2.0, dtype=float_type, name='sig')
         gp2 = LowRankApproxGP( n_components = 1000, random_state = rng)
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
    
    

  
    
    
