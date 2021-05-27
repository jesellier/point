# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:20:59 2021

@author: jesel
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float
import gpflow.kernels as gfk 

from point.low_rank.low_rank_base import LowRankBase
from point.utils import check_random_state_instance
from point.misc import Space



def expandedSum(x):
    z1 = tf.expand_dims(x, 1)
    z2 = tf.expand_dims(x, 0)

    return (z1 + z2, z1 - z2)



class LowRankRFF(LowRankBase):
    
    def __init__(self, kernel, beta0 = 1e-10, space = Space(), n_components = 1000, random_state = None):
       
        super().__init__(kernel, beta0, space, n_components, random_state)
        
        if not isinstance(kernel, gfk.SquaredExponential):
            raise NotImplementedError(" 'kernel' must of 'gfk.SquaredExponential' type")

        
    @property
    def lengthscales(self):
        return self.kernel.lengthscales
    
    @property
    def variance(self):
        return self.kernel.variance
    
    
    @property
    def trainable_variables_shape(self):
        lst =  []
        for v in self.trainable_variables :
            lst.append(v.numpy().shape[0])
        return lst
    
    
    def sample(self):
        random_state = check_random_state_instance(self._random_state)
        size = (self.n_features, self.n_components)

        self._G = tf.constant(random_state.normal(size = size), dtype=default_float(), name='G')
        self._random_offset = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype= default_float(), name='b')
        self._latent = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=default_float(), name='latent')


    def fit(self, sample = True):
        
        if sample : self.sample()

        gamma = 1 / (2 * self.lengthscales **2 )

        if len(gamma.shape) == 0 or gamma.shape[0] == 1 :
            self._random_weights =  tf.math.sqrt(2 * gamma) * self._G
            self._is_fitted = True
            return self

        self._random_weights =  tf.linalg.diag(tf.math.sqrt(2 * gamma))  @ self._G
        self._is_fitted = True
        return self
    
    
    def feature(self, X):
        """ Transforms the data X (n_samples, n_features) 
        to feature map space Z(X) (n_samples, n_components)"""
        
        if not self._is_fitted :
            raise ValueError("Random Fourrier object not fitted")
            
        if len(X.shape) == 1:
            n_features = X.shape[0]
            X = tf.reshape(X, (1, n_features))
        else :
            _, n_features = X.shape
            
        if n_features != self.n_features :
            raise ValueError("dimension of X must be =:" + str(self.n_features ))

        features = X @ self._random_weights  + self._random_offset
        features = tf.cos(features)
        features *= tf.sqrt(2 * self.variance / self.n_components)
     
        return features
    

    def __call__(self, X, X2 = None):
        if X2 is None :
            Z = self.feature(X)
            return Z @ tf.transpose(Z)
        return  self.feature(X)  @ tf.transpose(self.feature(X2))
 

    def func(self, X) :
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        features = self.feature(X)
        return features @ self._latent

    
    def integral(self, bounds = None):
        
        if bounds is None :
            bounds = self.space.bounds1D
            
        mat = self.__integral_mat(bounds)
        integral = tf.transpose(self._latent) @ mat @ self._latent
        integral += 2 * self.beta0 *  tf.transpose(self._latent) @ self.__integral_vec(bounds)
        integral += self.beta0**2  * self.space.measure
        integral = integral[0][0]

        return integral

    def maximum_log_likelihood_objective(self, X):

        int_term = self.integral()
        sum_term = sum(self.lambda_func(X))
            
        out = - int_term + sum_term
        
        return out
    
    
    def predict_f(self, Xnew):
        raise NotImplementedError


    def __integral_mat(self, bounds = [0,1]):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        up_bound = bounds[1]
        lo_bound = bounds[0]

        R =  self.n_components
        b = self._random_offset
        w = tf.transpose(self._random_weights)

        Mp, Mm = expandedSum(w)
        d1 = tf.stack([Mp[:,:,0] , tf.linalg.set_diag(Mm[:,:,0], tf.ones(R,  dtype=default_float())) ])
        d2 = tf.stack([Mp[:,:,1] , tf.linalg.set_diag(Mm[:,:,1], tf.ones(R,  dtype=default_float())) ])
    
        if b.shape == [] :
            b3 = tf.reshape(tf.stack([2 * b, tf.constant(0.0, dtype=default_float())]), shape = (2,1,1))
        else :
            (b1, b2) = expandedSum(b)
            b3 = tf.stack([b1, b2])
    
    
        M = tf.math.reduce_sum((1 / (d1 * d2)) * ( tf.cos(up_bound*d1 + lo_bound*d2 + b3) + tf.cos(lo_bound*d1 + up_bound*d2 + b3)  - tf.cos(up_bound*(d1 + d2) + b3)  - tf.cos(lo_bound*(d1 + d2) + b3)), axis = 0)

        diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (up_bound*w[:,0] + lo_bound* w[:,1] + b)) + tf.cos(2 * (lo_bound*w[:,0] + up_bound* w[:,1] + b)) \
                                                  - tf.cos(2 *up_bound* (w[:,0]+ w[:,1]) + 2 * b) - tf.cos(2 *lo_bound*(w[:,0]+ w[:,1]) + 2 * b) ) \
                                                  +  (up_bound - lo_bound)**2
        M = tf.linalg.set_diag(M, diag) 
    
        return self.variance * M / R
    
    
    
    
    def __integral_vec(self, bounds = [0,1]):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        up_bound = bounds[1]
        lo_bound = bounds[0]

        R =  self.n_components
        b = self._random_offset
        w = tf.transpose(self._random_weights)
        
        ws = w[:,0] + w[:,1]
        vec = tf.cos(up_bound*w[:,0] + lo_bound* w[:,1] + b) + tf.cos(lo_bound*w[:,0] + up_bound*w[:,1] + b)  \
                               - tf.cos(up_bound*(ws) + b)  - tf.cos(lo_bound*(ws) + b)
                               
        vec =  tf.linalg.diag(1 / (w[:,0] * w[:,1])) @ tf.expand_dims(vec, 1)
        vec *= tf.sqrt(tf.convert_to_tensor(2.0 * self.variance/ R, dtype=default_float()))

        return  vec
    

        

      
if __name__ == '__main__':
    rng = np.random.RandomState(20)

    beta0 = tf.Variable([0.5], dtype=default_float(), name='beta0')
    variance = tf.Variable([5], dtype=default_float(), name='sig')
    length_scale = tf.Variable([0.2,0.2], dtype=default_float(), name='lenghtscale')
    kernel = gfk.SquaredExponential(variance= variance , lengthscales= length_scale)

    lrgp = LowRankRFF(kernel, beta0 = beta0, n_components = 250, random_state = rng).fit()
    #print(lrgp.integral())
    
    X1 = tf.constant(rng.normal(size = [100, 2]), dtype=default_float(), name='X')
    X2 = tf.constant(rng.normal(size = [10, 2]), dtype=default_float(), name='X')

    K = kernel(X1).numpy()
    K2 = kernel(X1,X2).numpy()

    #print(lrgp.maximum_log_likelihood_objective(X))
    #print(lrgp.func(X))

    #lrgp.plot_kernel()
    #lrgp.plot_surface()
    
    Z = lrgp.feature(X1)
    test1 = Z @ tf.transpose(Z)
    test1 = test1.numpy()

    test2 = lrgp.feature(X1) @ tf.transpose(lrgp.feature(X2))
    test2 = test2.numpy()
    
    

    


    
    
    


        
        
        
        









    