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
import time

from point.helper import get_lrgp, method
from point.model import Space

from gpflow.config import default_float

import arrow

rng = np.random.RandomState()


def get_numerical_integral_sqrt(gp, offset = 0):
    bounds = gp.space.bounds1D
    integral_compare = integrate.dblquad( lambda x,y: (gp.func(tf.constant([x,y], dtype=default_float())) + offset)**2, 
                                         bounds[0], bounds[1], bounds[0], bounds[1])
    
    return integral_compare[0]


def get_numerical_integral(gp):
    bounds = gp.space.bounds1D
    integral_compare = integrate.dblquad( lambda x,y: (gp.func(tf.constant([x,y], dtype=default_float()))), 
                                         bounds[0], bounds[1], bounds[0], bounds[1])
    
    return integral_compare[0]


class Test_RFF_Integral(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.variance = tf.Variable(2.0, dtype=default_float(), name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=default_float(), name='l')
        self.offset = tf.Variable(1.0, dtype=default_float(), name='o')
        self.method = method.RFF
        self.space = Space([-1,1])
        
    
    #@unittest.SkipTest
    def test_integration_f_sqrt(self): 
        
        lrgp = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_sqrt(lrgp)
        
        print("")
        print("TEST RFF f.sqrt")
        print("integral.calculation:= : {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
    #@unittest.SkipTest
    def test_integration_f(self): 
        
        lrgp = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        
        bounds = lrgp.space.bounds1D
        integral_out = tf.transpose(lrgp._latent) @ lrgp._LowRankRFF__m(bounds)
        integral_out = (integral_out[0][0]).numpy()
        integral_recomputed = get_numerical_integral(lrgp)
        
        print("")
        print("TEST RFF f")
        print("integral.calculation:= : {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
    #@unittest.SkipTest
    def test_full_integration_with_drift(self): 
        
        lrgp = get_lrgp(method = self.method, variance = self.variance, beta0 = self.offset, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_sqrt(lrgp, self.offset.numpy())
        
        print("")
        print("TEST RFF full_with_drift")
        print("integral.calculation:= : {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
        

class Test_RFF_noOffset_Integral(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.variance = tf.Variable(2.0, dtype=default_float(), name='sig')
        self.length_scale = tf.Variable([0.1,0.1], dtype=default_float(), name='l')
        self.offset = tf.Variable(1.0, dtype=default_float(), name='o')
        self.method = method.RFF_NO_OFFSET
        self.space = Space([-1,1])
        
    #@unittest.SkipTest
    def test_integration_f_sqrt(self): 
        
        lrgp = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 75, random_state = rng)
        
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_sqrt(lrgp)
        
        print("")
        print("TEST RFF_no_offset f.sqrt ")
        print("integral.calculation:= {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
    #@unittest.SkipTest
    def test_integration_f(self): 

        lrgp = get_lrgp(method = self.method, variance = self.variance, beta0 = self.offset, length_scale = self.length_scale, space = self.space,
                    n_components = 75, random_state = rng)

        bounds = lrgp.space.bounds1D
        R =  lrgp.n_components
        cache = tf.linalg.diag(tf.ones(R,  dtype=default_float()))
        zeros = tf.zeros((R,R),  dtype=default_float())
        w1 = tf.experimental.numpy.hstack([cache , zeros]) @ lrgp._latent
        integral =  tf.transpose(w1) @ lrgp._LowRankRFFnoOffset__m(bounds)
        integral = (integral[0][0]).numpy()
        
        integral_recomputed = get_numerical_integral(lrgp)
        
        print("")
        print("TEST RFF_no_offset f ")
        print("integral.calculation:= : {}".format(integral))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral, integral_recomputed, places=7)

        
        



      
if __name__ == '__main__':
    unittest.main()
    
