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

from point.utils import check_random_state_instance
from point.random_fourier import RandomFourierWithOffset


rng = np.random.RandomState(40)


def repmat(vec, n):
    M =  tf.reshape(tf.tile(vec, tf.constant([n])), [n, tf.shape(vec)[0]])
    return M

def transformMat(vec, n):
    #for a vector vec return two matrices M1 = {v_i + v_j}_{i,j}  M1 = {v_i - v_j}_{i,j}
    M = repmat(vec, n)
    return (tf.transpose(M) + M, tf.transpose(M)-M)



class LowRankApproxGP():
    
    def __init__(self,  n_components = 1000, random_state = None):
        self.random_state = random_state
        self.n_components = n_components
        self.randomFourier_ = RandomFourierWithOffset(n_components = self.n_components, random_state = random_state)
        
        self._length_scale =  None
        self._variance = None
        self.is_fitted = False
        
        
        
    def trainable_variables(self):
        return {'variance' : self._variance, 'length_scale' : self._length_scale}
         

    def fit(self, length_scale, variance):
        random_state = check_random_state_instance(self.random_state)
        self._length_scale =  length_scale
        self._variance = variance
        
        #sample the betas
        size = (self.n_components, 1)
        self.beta_ = tf.constant(random_state.normal(size = size), dtype=float_type, name='beta')

        #fit the RFF
        gamma = 1 / (2 * self._length_scale **2 )
        self.randomFourier_.fit(gamma = gamma, variance = self._variance, )
        self.is_fitted = True
        
        return self
        
    
    def reset_trainable_variables(self, variance, length_scale):
        self._variance = variance
        self._length_scale = length_scale 
        gamma = 1 / (2 * self._length_scale **2 )
        self.randomFourier_.reset_trainable_variables(self._variance, gamma)


    def func(self, X) :
        if not self.is_fitted :
            raise ValueError("instance not fitted")

        features = self.randomFourier_.feature(X)
        return features @ self.beta_

    
    def integral_grad(self, length_scale = None, variance = None, T = 1.0):
        
        if length_scale is None :
            length_scale = self._length_scale
            
        if variance is None :
            variance = self._variance
            
        with tf.GradientTape() as tape:      
            self.reset_trainable_variables(variance, length_scale)
            mat = self.__integral_mat(T)
            out = tf.transpose(self.beta_) @ mat @ self.beta_
            out = out[0][0]
            
        grad = tape.gradient(out, [variance, length_scale])
        return (out, grad)
    
    
    def likelihood_grad(self, X, length_scale = None, variance = None, T = 1.0):
        
        if length_scale is None :
            length_scale = self._length_scale
            
        if variance is None :
            variance = self._variance

        with tf.GradientTape() as tape: 
            self.reset_trainable_variables(variance, length_scale)
            mat = self.__integral_mat(T)
            
            int_term = tf.transpose(self.beta_) @ mat @ self.beta_
            int_term = int_term[0][0]
            
            f = self.func(X)
            sum_term = tf.norm(f)
            sum_term = tf.math.square(sum_term)
            
            out = int_term + sum_term
    
        grad = tape.gradient(out, [variance, length_scale] )
        return (out, grad)

        

    
    def __integral_mat(self, T = 1.0):
        """ w = vector of weights, T = expiry, b = vector of drifts """
        if not self.is_fitted :
            raise ValueError("instance not fitted")

        R =  self.randomFourier_.n_components
        w = tf.transpose(self.randomFourier_.random_weights_)
        b = self.randomFourier_.random_offset_
        (A, C) = transformMat(w[:, 0], R)
        (B, D) = transformMat(w[:, 1], R)
        C = tf.linalg.set_diag(C, tf.ones(R,  dtype=float_type)) 
        D = tf.linalg.set_diag(D, tf.ones(R,  dtype=float_type)) 
    
        if b.shape == [] :
            b1 = 2 * b
            b2 = tf.constant(0.0, dtype=float_type)
        else :
            (b1, b2) = transformMat(b, R)

        mat = (1 / (A * B)) * ( tf.cos(T * A + b1) + tf.cos(T * B + b1) - tf.cos(T * (A + B) + b1)  - tf.cos(b1))
        mat += (1 / (C * D)) * ( tf.cos(T * C + b2) + tf.cos(T * D + b2) - tf.cos(T * (C + D) + b2)  - tf.cos(b2))
        
        bdo = 2 * b
        diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 *T * w[:,1] + bdo) + tf.cos(2 *T * w[:,0] + bdo) - tf.cos(2 *T * ( w[:,0] +  w[:,1] ) + bdo)  - tf.cos(bdo)) +  T**2
        mat = tf.linalg.set_diag(mat, diag) 
    
        return  mat / R
        
        

      
if __name__ == '__main__':
    rng = np.random.RandomState(40)
    X = tf.constant(rng.normal(size = [100, 2]), dtype=float_type, name='X')
    
    variance = tf.Variable(0.5, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='lenght_scale')

    gp = LowRankApproxGP(n_components = 1000, random_state = rng).fit(length_scale, variance)
    out, grad = gp.integral_grad()
    out, grad = gp.likelihood_grad(X)
    print(out)
    print(grad)
    
    
    


        
        
        
        









    