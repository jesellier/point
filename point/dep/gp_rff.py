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

from point.utils import check_random_state_instance, transformMat
from point.random_fourier import RandomFourierWithOffset
    
from mpl_toolkits.mplot3d import Axes3D  # Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt

rng = np.random.RandomState(40)




class LowRankApproxGP():
    
    def __init__(self,  n_components = 1000, random_state = None):
        self.random_state = random_state
        self.n_components = n_components
        self.randomFourier_ = RandomFourierWithOffset(n_components = self.n_components, random_state = random_state)
        
        self.transformer = None
        self._length_scale =  None
        self._variance = None
        self.is_fitted = False
        

    def trainable_variables(self):
        return {'variance' : self._variance, 'length_scale' : self._length_scale}
         

    def fit(self, length_scale, variance, transformer = None):
        random_state = check_random_state_instance(self.random_state)
        self._length_scale =  length_scale
        self._variance = variance

        #sample the betas
        size = (self.n_components, 1)
        self.beta_ = tf.constant(random_state.normal(size = size), dtype=float_type, name='beta')

        #fit the RFF
        gamma = 1 / (2 * self._length_scale **2 )
        self.randomFourier_.fit(gamma = gamma, variance = self._variance)
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

    
    def integral_grad(self, lplus = 1.0, lminus = 0):

        length_scale = self._length_scale
        variance = self._variance
  
        with tf.GradientTape() as tape:      
            self.reset_trainable_variables(variance, length_scale)
            mat = self.__integral_mat(lplus, lminus)
            out = tf.transpose(self.beta_) @ mat @ self.beta_
            out = out[0][0]

        grad = tape.gradient(out, [variance, length_scale])
        return (out, grad)
    
    
    def likelihood_grad(self, X, lplus = 1.0, lminus = 0):

        length_scale = self._length_scale
        variance = self._variance

        with tf.GradientTape() as tape: 
            self.reset_trainable_variables(variance, length_scale)
            mat = self.__integral_mat(lplus, lminus)

            int_term = tf.transpose(self.beta_) @ mat @ self.beta_
            int_term = int_term[0][0]
            
            f = self.func(X)
            sum_term = tf.norm(f)
            sum_term = tf.math.square(sum_term)
            
            out = int_term + sum_term
    
        grad = tape.gradient(out, [variance, length_scale] )
        return (out, grad)

        

    
    def __integral_mat(self, lplus = 1.0, lminus = 0):
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
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.show()
        pass
        

        

      
if __name__ == '__main__':
    rng = np.random.RandomState()

    variance = tf.Variable(2, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.5,100], dtype=float_type, name='lenght_scale')

    gp = LowRankApproxGP(n_components = 100, random_state = rng).fit(length_scale, variance).fit(length_scale, variance )
    #gp.plot_surface()
    #K = gp.randomFourier_.kernel(inputs).numpy()
    
    out, grad = gp.integral_grad()
    
    #TEST
    with tf.GradientTape() as tape:  
            gamma = 1 / (2 * gp._length_scale **2 )
            gp.randomFourier_.reset_trainable_variables(gp._variance, gamma)
            w = tf.transpose(gp.randomFourier_.random_weights_)

    grad = tape.gradient(w, gp._length_scale)

    
    
    
    


    
    
    


        
        
        
        









    