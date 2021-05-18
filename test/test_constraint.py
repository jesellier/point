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

import unittest

from gpflow.base import Parameter
from gpflow.utilities import positive

rng = np.random.RandomState(40)

class Struct():
    
    def __init__(self, lengthscales , variance):
        self.lengthscales = Parameter(lengthscales, transform=positive())
        self.variance = Parameter(variance, transform=positive())

class Test_Integration(unittest.TestCase):

    def setUp(self):
        v = tf.Variable(1.0, dtype=float_type, name='sig')
        
        # by default set the softmax positive i.e v = log(1 + exp( v_unconstrained ))
        # i.e. for v = 1, v_un = 0.5413248546129181
        self.param = Parameter(v, transform=positive())  
        
    def test_compare_integration1(self): 

        param = self.param
        
        with tf.GradientTape() as tape:
            out = 2 * param + 1
            
        # grad = dv/dv_un = exp(v_un) / (1 + exp v_un) = 1.2642411176571153
        grad = tape.gradient(out, param.trainable_variables) 
        
        
        # update the trainable variable i.e v_un = v_un - grad = -0.7229162630441972
        optimizer = tf.keras.optimizers.SGD(learning_rate= 1.0 )
        optimizer.apply_gradients(zip(grad, param.trainable_variables))
        
        # the param should be synchronized 0.39564021899309176

      
if __name__ == '__main__':
    unittest.main()
    
    

  
    
    
