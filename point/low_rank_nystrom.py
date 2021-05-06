# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np
#import scipy as scp

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

import gpflow.kernels as gfk

#from sklearn.gaussian_process.kernels import RBF
from point.utils import check_random_state_instance
from point.low_rank_base import LowRankBase
from point.misc import Space


from enum import Enum


class LowRankNystrom(LowRankBase):
    
    class mode(Enum):
        SAMPLING = 1
        GRID = 2


    def __init__(self, kernel, space = Space(), n_components = 250, random_state = None, noise = 1e-5, mode = 'grid'):
        
        super().__init__(space, n_components, random_state)

        self._noise = noise
        self._impl_kernel = kernel
        
        if mode == 'sampling' :
            self.mode = LowRankNystrom.mode.SAMPLING
        else :
            self.mode = LowRankNystrom.mode.GRID
            

    @property
    def trainable_variables(self):
        return self._impl_kernel.trainable_variables
    
    @property
    def parameters(self):
        return self._impl_kernel.parameters

        
    def fit(self, sample = True):
        if sample : self.sample()

        self.__evd()
        self._vl =  self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda)) @ self.latent_
        self._is_fitted = True
        
        return self

    def sample(self):
        random_state = check_random_state_instance(self._random_state)
        self.latent_ = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=float_type, name='beta')
        
        if self.mode == LowRankNystrom.mode.SAMPLING :
            self.__sample_x()
        elif self.mode == LowRankNystrom.mode.GRID :
            self.__grid_x()
        else :
            raise ValueError("Mode not recognized")
  
        self.fit(sample = False)

   
    def __sample_x(self):
        random_state = check_random_state_instance(self.random_state)
        bounds = self.space.bounds
        sample = tf.constant(random_state.uniform(bounds[:, 0], bounds[:, 1], size=(self.n_components, self.n_features)), 
                             dtype=float_type, 
                             name='x')
        
        self._x = sample

        
    def __grid_x(self):
        
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bounds1D
        
        step = 1/np.sqrt(self.n_components)
        x = np.arange(bounds[0], bounds[1], step)
        y = np.arange(bounds[0], bounds[1], step)
        X, Y = np.meshgrid(x, y)
        
        n_elements = X.shape[0]**2
        inputs = np.zeros((n_elements ,2))
        inputs[:,0] = np.ravel(X)
        inputs[:,1] = np.ravel(Y)
        
        if n_elements != self.n_components :
            shuffled_idx = np.arange(n_elements)
            random_state.shuffle(shuffled_idx)
            shuffled_idx = shuffled_idx[- self.n_components :]
            inputs = inputs[shuffled_idx]
        
        sample = tf.constant(inputs, dtype=float_type, name='x')
        self._x = sample
        
        
        
    def __evd(self):
        K = self._impl_kernel(self._x, self._x)
        K = K + tf.eye(K.shape[0], dtype=float_type) * tf.constant(self._noise, dtype=float_type) 
        self._lambda, self._U, V = tf.linalg.svd(K)
        
        
    def __validate_entry(self, X):
        if len(X.shape) == 1:
            n_features = X.shape[0]
            X = tf.reshape(X, (1, n_features))
        else :
            _, n_features = X.shape
        return X


    def inv(self):
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        return self._U @ tf.linalg.diag(1/self._lambda) @ tf.transpose(self._U)
    
    
    def feature(self, X):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
        
        X = self.__validate_entry(X)
        out = self._impl_kernel(X, self._x) @ self._U @ tf.linalg.diag(1/self._lambda)

        return out
    
    
    def func(self, X) :
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        
        X = self.__validate_entry(X)
        return self._impl_kernel(X, self._x) @ self._vl


    def kernel(self, X):
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        X = self.__validate_entry(X)
        K = self._impl_kernel(X, self._x)
        return K @ self.inv() @ tf.transpose(K)
    

    def integral(self, bounds = None):
        out = tf.transpose(self.latent_) @ tf.linalg.diag(tf.math.sqrt(self._lambda)) @ self.latent_
        out =  (self.space.measure / self.n_components) * out
        out = out[0][0]
        return out
        
    
    def likelihood(self, X, bounds = None):
        
        if bounds is None :
            bounds = self.space.bounds
        
        int_term = self.integral(bounds)

        f = self.func(X)
        sum_term = tf.norm(f)
        sum_term = tf.math.square(sum_term)
            
        out = sum_term - int_term
        
        return out

    
    
    
if __name__ == "__main__":

    rng  = np.random.RandomState(10)
    variance = tf.Variable(5, dtype=float_type, name='sig')
    length_scale = tf.Variable(0.5, dtype=float_type, name='lenght_scale')

    X = tf.constant(rng.normal(size = [500, 2]), dtype=float_type, name='X')
    kernel = gfk.SquaredExponential(variance= variance , lengthscales= length_scale)
    lrgp = LowRankNystrom(kernel, n_components = 250, random_state=rng, mode = 'grid').fit()
    
   # print(lrgp.integral())
    #print(lrgp.parameters)
    print(lrgp.likelihood(X))
    
    bounds = lrgp.space.bounds
    int_term = lrgp.integral(bounds)

    f = lrgp.func(X)
    sum_term = tf.norm(f)
    sum_term = tf.math.square(sum_term)
            
    out = sum_term - int_term
    
    #lrgp.plot_kernel()
    #lrgp.plot_surface()


    
    # ################
    # Kxx2 = kernel2(x,x)
    # Kxx2[np.diag_indices_from(Kxx2)] += noise
    
    # U2, s2, V2  = scp.linalg.svd(Kxx2)
    # v2  =  np.multiply(U2, np.sqrt(1/s2))
    # inv2 = v2 @ v2.T
    # test2 = inv2 @ Kxx2
    
    # eps = eigvalsh_to_eps(s2, None, None)
    # if np.min(s2) < -eps:
    #     raise ValueError('the input matrix must be positive semidefinite')
    # d = s2[s2 > eps]
    # if len(d) < len(s2) :
    #     raise np.linalg.LinAlgError('singular matrix')
        
    # Kx2 = kernel2(X, x)
    # K2 = Kx2 @ inv2 @ Kx2.T
    
    

    

    
