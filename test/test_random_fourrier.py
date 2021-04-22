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
rng = np.random.RandomState(40)

import gpflow
from sklearn.gaussian_process.kernels import RBF

import unittest

from point.random_fourier import RandomFourier, RandomFourierWithOffset


    
def exponentialKernel(x1, x2, lengthscales):
    gamma = 1 / (2 * lengthscales **2 )
    diff = tf.square(x1 - x2)
    k = tf.exp(- (diff[0] * gamma[0] + diff[1]  * gamma[1] ))
    return k


class Test_RFF(unittest.TestCase):
    
    
    def setUp(self):
        
        self.size = 10
        self.X = tf.constant(rng.normal(size = [self.size, 2]), dtype=float_type, name='X')
        
        self.length_scale = tf.Variable([0.2,2], dtype=float_type, name='l')
        self.gamma = 1 / (2 * self.length_scale **2 )
        self.unit_variance = tf.Variable(1.0, dtype=float_type, name='sig')
    
    
    def test_rescaling_rff(self):
        "verify the scaling of the variance term in the feature vector"
        
        size = self.size
        X = self.X
        gamma = self.gamma
        unit_variance = self.unit_variance
        
        rbf1 = RandomFourier(n_components=size, gamma = gamma, variance = unit_variance, random_state = rng).sample()
        feature1 = rbf1.feature(X).numpy()
        k1 = rbf1.kernel(X).numpy()

        variance = tf.Variable(2.0, dtype=float_type, name='sig')
        rbf2 = RandomFourier(n_components=size, gamma = self.gamma, variance = variance, random_state = rng).sample()
        rbf2.random_weights_ = rbf1.random_weights_
        
        feature2 = rbf2.feature(X).numpy()
        k2 = rbf2.kernel(X).numpy()
        
        variance  = variance.numpy()
        rescaled_feature1 =  np.sqrt(variance) * feature1
        rescaled_k1 = variance * k1
        
        for i in range(size):
            for j in range(size):
                self.assertAlmostEqual(rescaled_feature1[i][j], feature2[i][j], places=7)
                self.assertAlmostEqual(rescaled_k1[i][j], k2[i][j], places=7)
              
                
              
    def test_rescaling_rff_withoffset(self):
        "verify the scaling of the variance term in the feature vector"
        
        size = self.size
        X = self.X
        gamma = self.gamma
        unit_variance = self.unit_variance
        
        rbf1 =  RandomFourierWithOffset(n_components=size, gamma = gamma, variance = unit_variance, random_state = rng).sample()
        feature1 = rbf1.feature(X).numpy()
        k1 = rbf1.kernel(X).numpy()
    
        variance = tf.Variable(2.0, dtype=float_type, name='sig')
        rbf2 =  RandomFourierWithOffset(n_components=size, gamma = gamma, variance = variance, random_state = rng).sample()
        rbf2.random_weights_ = rbf1.random_weights_
        rbf2.random_offset_ = rbf1.random_offset_
        
        feature2 = rbf2.feature(X).numpy()
        k2 = rbf2.kernel(X).numpy()
        
        variance  = variance.numpy()
        rescaled_feature1 =  np.sqrt(variance) * feature1
        rescaled_k1 = variance * k1
        
        for i in range(size):
            for j in range(size):
                self.assertAlmostEqual(rescaled_feature1[i][j], feature2[i][j], places=7)
                self.assertAlmostEqual(rescaled_k1[i][j], k2[i][j], places=7)
    
    
        


if __name__ == '__main__':
    unittest.main()

    X = tf.constant(rng.normal(size = [1000, 2]), dtype=float_type, name='X')
    amplitude = tf.Variable(5.0, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.2,2], dtype=float_type, name='l')

    ## Kernel in GPflow
    kernelg = gpflow.kernels.SquaredExponential(variance= amplitude **2 , lengthscales= length_scale)
    Kg = kernelg(X,X).numpy()
    #K = K + tf.eye(K.shape[0], dtype=float_type) *  tf.constant(noise, dtype=float_type) 

    ## Kernel in sklearn
    kernels = (amplitude**2) * RBF(length_scale= length_scale)
    Ks = kernels(X)
    
    ## RF
    gamma = 1 / (2 * length_scale **2 )
    rbf1 = RandomFourier(n_components=1000, gamma = gamma, variance = amplitude**2, random_state = rng).sample()
    rbf2 = RandomFourierWithOffset(n_components=1000, gamma = gamma, variance = amplitude**2 , random_state = rng).sample()
    test = exponentialKernel(X[0], X[1], gamma)
    
    K1 = rbf1.kernel(X).numpy()
    K2 = rbf2.kernel(X).numpy()

    error1 = sum(sum((Kg - K1)**2))
    error2 = sum(sum((Kg - K2)**2))
    
    with tf.GradientTape() as tape: 
         out = amplitude * exponentialKernel(X[0], X[1], length_scale)
         #print(out)

    grad1 = tape.gradient(out, length_scale )
    #print(grad1.numpy())
    

    with tf.GradientTape() as tape: 
         g = 1 / (2 * length_scale **2 )
         rbf_grad = RandomFourier(n_components=100, gamma = g, random_state = rng).sample()
         out = rbf_grad.kernel(X)[0,1]
         #print(out)

    grad2 = tape.gradient(out, length_scale ).numpy()
    #print(grad2)
    
    