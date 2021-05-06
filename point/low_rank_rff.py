# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:20:59 2021

@author: jesel
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

float_type = tf.dtypes.float64

from gpflow.base import Parameter
from gpflow.utilities import positive

from point.low_rank_base import LowRankBase
from point.utils import check_random_state_instance, transformMat
from point.misc import Space

rng = np.random.RandomState(40)



class LowRankRFF(LowRankBase):
    
    def __init__(self, length_scale, variance, space = Space(), n_components = 1000, random_state = None):
       
        super().__init__(space, n_components, random_state)

        self._length_scale = Parameter(length_scale , transform=positive())
        self._variance =  Parameter(variance, transform=positive())


    @property
    def parameters(self):
        return (self._length_scale , self._variance)
        
        
    @property
    def trainable_variables(self):
        l = []
        if self._length_scale.trainable:
            l.append(self._length_scale.trainable_variables[0])
        if self._variance.trainable :
            l.append(self._variance.trainable_variables[0])
            
        #return (self._length_scale , self._variance)
        return tuple(l)
    
    #@property
    #def trainable_variables_shape(self):
        #l = []
        
        



    def sample(self):
        random_state = check_random_state_instance(self._random_state)
        size = (self.n_features, self.n_components)

        self._z = tf.constant(random_state.normal(size = size), dtype=float_type, name='z')
        self._random_offset = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype=float_type, name='b')
        self._beta = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=float_type, name='beta')


    def fit(self, sample = True):
        
        if sample : self.sample()

        gamma = 1 / (2 * self._length_scale **2 )

        if len(gamma.shape) == 0 or gamma.shape[0] == 1 :
            self._random_weights =  tf.math.sqrt(2 * gamma) * self._z
            self._is_fitted = True
            return self

        self._random_weights =  tf.linalg.diag(tf.math.sqrt(2 * gamma))  @ self._z
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

        output = X @ self._random_weights  + self._random_offset
        output = tf.cos(output)
        output = tf.sqrt(2 * self._variance /tf.constant(self.n_components, dtype=float_type)) * output
     
        return output
    

    def kernel(self, X):
        Z = self.feature(X)
        return Z @ tf.transpose(Z)
 

    def func(self, X) :
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        features = self.feature(X)
        return features @ self._beta

    
    def integral(self, bounds = None):
        
        if bounds is None :
            bounds = self.space.bounds1D

        mat = self.__integral_mat(bounds)
        out = tf.transpose(self._beta) @ mat @ self._beta
        out = out[0][0]

        return out
    
    
    def likelihood(self, X, bounds = None):
        
        if bounds is None :
            bounds = self.space.bounds1D
 
        mat = self.__integral_mat(bounds)

        int_term = tf.transpose(self._beta) @ mat @ self._beta
        int_term = int_term[0][0]
            
        f = self.func(X)
        sum_term = tf.norm(f)
        sum_term = tf.math.square(sum_term)
            
        out = - int_term + sum_term
        
        return out


    
    def __integral_mat(self, bounds = [0,1]):
        """ w = vector of weights, T = expiry, b = vector of drifts """
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        Upbound = bounds[1]
        Lowbound = bounds[0]

        R =  self.n_components
        w = tf.transpose(self._random_weights)
        b = self._random_offset
        
        (A, C) = transformMat(w[:, 0], R)
        (B, D) = transformMat(w[:, 1], R)
        C = tf.linalg.set_diag(C, tf.ones(R,  dtype=float_type)) 
        D = tf.linalg.set_diag(D, tf.ones(R,  dtype=float_type)) 
    
        if b.shape == [] :
            b1 = 2 * b
            b2 = tf.constant(0.0, dtype=float_type)
        else :
            (b1, b2) = transformMat(b, R)

        mat = (1 / (A * B)) * ( tf.cos(Upbound*A + Lowbound*B + b1) + tf.cos(Lowbound*A + Upbound*B + b1)  \
                               - tf.cos(Upbound*(A + B) + b1)  - tf.cos(Lowbound*(A + B) + b1))
        mat += (1 / (C * D)) * (tf.cos(Upbound*C + Lowbound*D + b2) + tf.cos(Lowbound*C + Upbound*D + b2)  \
                                - tf.cos(Upbound*(C + D) + b2)  - tf.cos(Lowbound*(C + D) + b2 ))
        
        bdo = 2 * b
        diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (Upbound*w[:,0] + Lowbound* w[:,1]) + bdo) + tf.cos(2 * (Lowbound*w[:,0] + Upbound* w[:,1]) + bdo) \
                                              - tf.cos(2 *Upbound* (w[:,0]+ w[:,1]) + bdo) - tf.cos(2 *Lowbound*(w[:,0]+ w[:,1]) + bdo) ) \
                                              +  (Upbound - Lowbound)**2
        mat = tf.linalg.set_diag(mat, diag) 
    
        return  self._variance * mat / R
    

        

      
if __name__ == '__main__':
    rng = np.random.RandomState(20)

    variance = tf.Variable(5, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='lenght_scale')

    lrgp = LowRankRFF(length_scale, variance, n_components = 250, random_state = rng)
    lrgp.fit()
    #print(lrgp.integral())
    
    X = tf.constant(rng.normal(size = [10, 2]), dtype=float_type, name='X')
    #print(lrgp.likelihood(X))
    print(lrgp.func(X))

    lrgp.plot_kernel()
    lrgp.plot_surface()



    
    
    


        
        
        
        









    