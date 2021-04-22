# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:20:59 2021

@author: jesel
"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

import warnings
from point.utils import check_random_state_instance


class RandomFourrierBase():
    
    def __init__(self, n_components=100, n_features = 2, gamma= None, variance = 1.0, random_state=None):
        self.n_components = n_components
        self.n_features = n_features
        self.gamma = gamma
        self.variance = variance
        self.random_state = random_state
        self.is_fitted = False


    def kernel(self, X):
        Z = self.feature(X)
        return Z @ tf.transpose(Z)

    
class RandomFourier(RandomFourrierBase):
    """ assume G.kernel k(x) = exp( - gamma * ||x||**2)   i.e. gamma = 1 / (2 * sig ** 2)
       thus gamma = m <=> sig = sqrt(1/2 * m) = 1/sqrt(2 * m)
       thus for w ~ N(0,1/sig) <=> w = sqrt(2 * m) * Z """
 
    def __init__(self, n_components=100, n_features = 2, gamma= None, variance = 1.0, random_state=None):
        super().__init__(n_components, n_features, gamma, variance, random_state)
        
        
    def sample(self):
        
        random_state = check_random_state_instance(self.random_state)
        n_features = self.n_features
        n_components = int(self.n_components / 2)

        if self.n_components % 2 != 0:
            self.n_components = 2 * n_components
            warnings.warn("n_components % 2 != 0. n_components is changed "
                              " to {}.".format(self.n_components))
        if self.gamma is None :
            gamma =  tf.Variable(tf.ones(n_features,  dtype=float_type), dtype=float_type)
        elif self.gamma.shape == [] :
            gamma = self.gamma * tf.Variable(tf.ones(n_features,  dtype=float_type), dtype=float_type)
        else :
            gamma = self.gamma

        size = (n_features, n_components)
        std = tf.linalg.diag(tf.math.sqrt(2 * gamma)) 
        self.random_weights_=  std @ tf.constant(random_state.normal(size = size), dtype=float_type, name='z')
        self.is_fitted = True

        return self
    
    def feature(self, X):
        """transform the random weights int orandom features """
        if len(X.shape) == 1:
            n_features = X.shape[0]
            X = tf.reshape(X, (1, n_features))
        else :
            _, n_features = X.shape
            
        if n_features != self.n_features :
            raise ValueError("dimension of X must be =:" + str(self.n_features ))

        output = X @ self.random_weights_
        output = tf.concat((tf.cos(output), tf.sin(output)), axis = 1)
        output = tf.sqrt(2 * self.variance /tf.constant(self.n_components, dtype=float_type)) * output

        return output
    



class RandomFourierWithOffset(RandomFourrierBase) :
    """ assume G.kernel k(x) = exp( - gamma * ||x||**2)   i.e. gamma = 1 / (2 * sig ** 2)
       thus gamma = m <=> sig = sqrt(1/2 * m) = 1/sqrt(2 * m)
       thus for w ~ N(0,1/sig) <=> w = sqrt(2 * m) * Z """

 
    def __init__(self, n_components=100, n_features = 2, gamma= None, variance = 1.0, random_state=None):
        super().__init__(n_components, n_features, gamma, variance, random_state)
        
        
    def sample(self):
        
        random_state = check_random_state_instance(self.random_state)
        n_components = self.n_components
        n_features = self.n_features
        
        if self.gamma is None :
            gamma =  tf.Variable(tf.ones(n_features,  dtype=float_type), dtype=float_type)
        elif self.gamma.shape == [] :
            gamma = self.gamma * tf.Variable(tf.ones(n_features,  dtype=float_type), dtype=float_type)
        else :
            gamma = self.gamma
            
        size = (n_features, n_components)
        std = tf.linalg.diag(tf.math.sqrt(2 * gamma)) 
        self.random_weights_=  std @ tf.constant(random_state.normal(size = size), dtype=float_type, name='z')
        self.random_offset_ = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype=float_type, name='b')
        self.is_fitted = True
        
        return self
    
    
    def feature(self, X):
        """ Transforms the data X (n_samples, n_features) to feature map space Z(X) (n_samples, n_components)"""
        if not self.is_fitted :
            raise ValueError("Random Fourrier object not fitted")
            
        if len(X.shape) == 1:
            n_features = X.shape[0]
            X = tf.reshape(X, (1, n_features))
        else :
            _, n_features = X.shape
            
        if n_features != self.n_features :
            raise ValueError("dimension of X must be =:" + str(self.n_features ))


        output = X @ self.random_weights_
        output += self.random_offset_
        output = tf.cos(output)
        output = tf.sqrt(2 * self.variance /tf.constant(self.n_components, dtype=float_type)) * output
     
        return output
