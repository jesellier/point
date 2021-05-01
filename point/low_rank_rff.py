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

import gpflow
from gpflow.utilities import positive

from point.utils import check_random_state_instance, transformMat
import matplotlib.pyplot as plt

from gpflow.base import Parameter
from gpflow.utilities import positive


rng = np.random.RandomState(40)



class LowRankRFF():
    
    def __init__(self, length_scale, variance, n_components = 1000, random_state = None):
        self.random_state = random_state
        
        self.n_components = n_components
        self.n_features = 2
        
        self._length_scale = Parameter(length_scale , transform=positive())
        self._variance =  Parameter(variance, transform=positive())
        #self._length_scale = Parameter(length_scale)
        #self._variance =  Parameter(variance)
        #self._length_scale = length_scale
        #self._variance =  variance
        
    
    @property
    def parameters(self):
        return (self._length_scale , self._variance)
        
        
    @property
    def trainable_variables(self):
        #return (self._length_scale , self._variance)
        return (self._length_scale.trainable_variables[0] , self._variance.trainable_variables[0])
    
        
    
    def sample(self):
        random_state = check_random_state_instance(self.random_state)
        size = (self.n_features, self.n_components)

        self.z_ = tf.constant(random_state.normal(size = size), dtype=float_type, name='z')
        self.random_offset_ = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype=float_type, name='b')
        self.beta_ = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=float_type, name='beta')


    def fit(self, sample = True):
        
        if sample : self.sample()

        #self._length_scale, self._variance =  trainable_variables
        gamma = 1 / (2 * self._length_scale **2 )
        self.random_weights_=  tf.linalg.diag(tf.math.sqrt(2 * gamma))  @ self.z_
        self.is_fitted = True
        
        return self
    
    
    def feature(self, X):
        """ Transforms the data X (n_samples, n_features) 
        to feature map space Z(X) (n_samples, n_components)"""
        
        if not self.is_fitted :
            raise ValueError("Random Fourrier object not fitted")
            
        if len(X.shape) == 1:
            n_features = X.shape[0]
            X = tf.reshape(X, (1, n_features))
        else :
            _, n_features = X.shape
            
        if n_features != self.n_features :
            raise ValueError("dimension of X must be =:" + str(self.n_features ))

        output = X @ self.random_weights_  + self.random_offset_
        output = tf.cos(output)
        output = tf.sqrt(2 * self._variance /tf.constant(self.n_components, dtype=float_type)) * output
     
        return output
    

    def kernel(self, X):
        Z = self.feature(X)
        return Z @ tf.transpose(Z)
 

    def func(self, X) :
        if not self.is_fitted :
            raise ValueError("instance not fitted")
        features = self.feature(X)
        return features @ self.beta_

    
    def integral(self, lplus = 1.0, lminus = 0):
        #self.fit((self._length_scale, self._variance), sample = False)
        mat = self.__integral_mat(lplus, lminus)
        out = tf.transpose(self.beta_) @ mat @ self.beta_
        out = out[0][0]

        return out
    
    
    def likelihood(self, X, lplus = 1.0, lminus = 0):
        #self.fit((self._length_scale, self._variance), sample = False)
        mat = self.__integral_mat(lplus, lminus)

        int_term = tf.transpose(self.beta_) @ mat @ self.beta_
        int_term = int_term[0][0]
            
        f = self.func(X)
        sum_term = tf.norm(f)
        sum_term = tf.math.square(sum_term)
            
        out = - int_term + sum_term
        
        return out


    
    def __integral_mat(self, lplus = 1.0, lminus = 0):
        """ w = vector of weights, T = expiry, b = vector of drifts """
        if not self.is_fitted :
            raise ValueError("instance not fitted")

        R =  self.n_components
        w = tf.transpose(self.random_weights_)
        b = self.random_offset_
        
        (A, C) = transformMat(w[:, 0], R)
        (B, D) = transformMat(w[:, 1], R)
        C = tf.linalg.set_diag(C, tf.ones(R,  dtype=float_type)) 
        D = tf.linalg.set_diag(D, tf.ones(R,  dtype=float_type)) 
    
        if b.shape == [] :
            b1 = 2 * b
            b2 = tf.constant(0.0, dtype=float_type)
        else :
            (b1, b2) = transformMat(b, R)

        mat = (1 / (A * B)) * ( tf.cos(lplus*A + lminus*B + b1) + tf.cos(lminus*A + lplus*B + b1)  \
                               - tf.cos(lplus*(A + B) + b1)  - tf.cos(lminus*(A + B) + b1))
        mat += (1 / (C * D)) * (tf.cos(lplus*C + lminus*D + b2) + tf.cos(lminus*C + lplus*D + b2)  \
                                - tf.cos(lplus*(C + D) + b2)  - tf.cos(lminus*(C + D) + b2 ))
        
        bdo = 2 * b
        diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (lplus*w[:,0] + lminus* w[:,1]) + bdo) + tf.cos(2 * (lminus*w[:,0] + lplus* w[:,1]) + bdo) \
                                              - tf.cos(2 *lplus* (w[:,0]+ w[:,1]) + bdo) - tf.cos(2 *lminus*(w[:,0]+ w[:,1]) + bdo) ) \
                                              +  (lplus - lminus)**2
        mat = tf.linalg.set_diag(mat, diag) 
    
        return  self._variance * mat / R
    


    
    def plot_surface(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-1.0, 1.0, 0.05)
        X, Y = np.meshgrid(x, y)
    
        n = X.shape[0]**2
        inputs = np.zeros((n,2))
        inputs[:,0] = np.ravel(X)
        inputs[:,1] = np.ravel(Y)
        
        zs = np.array(self.func(X = inputs)**2)
        Z = zs.reshape(X.shape)
    
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Instensity')
        
        plt.show()
        pass
        
        

      
if __name__ == '__main__':
    rng = np.random.RandomState()

    variance = tf.Variable(5, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='lenght_scale')

    gp = LowRankRFF(length_scale, variance, n_components = 250, random_state = rng)
    gp.fit()
   
    gp.plot_surface()

    #TEST
    # with tf.GradientTape() as tape:  
    #     out = gp.integral()
    # grad = tape.gradient(out, gp._length_scale)

    
    
    
    


    
    
    


        
        
        
        









    