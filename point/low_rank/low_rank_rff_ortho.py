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

from point.low_rank.low_rank_rff_with_offset import LowRankRFF
from point.low_rank.low_rank_rff_no_offset import LowRankRFFnoOffset
from point.utils import check_random_state_instance
from point.misc import Space

#from scipy.linalg import qr_multiply
from scipy.stats import chi



class LowRankRFFOrthogonal(LowRankRFF):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, n_features = 1, random_state = None):
       
        super().__init__(kernel, beta0, space, n_components, n_features, 
                         random_state)
        
        self.n_stacks = int(np.ceil(self.n_components/self.n_features))
        self.n_components = self.n_stacks * self.n_features
        
    
    def set_points_trainable(self, trainable):
        
        if not self._is_fitted :
            raise ValueError("Random Fourrier object not fitted")

        if trainable is True :
            self._points_trainable = True
            self._offset_trainable = True
            self._W = tf.Variable(self._W)
            self._offset = tf.Variable(self._offset)
        else :
            self._points_trainable = False
            self._offset_trainable = False
            self._W = tf.constant(self._W)
            self._offset = tf.constant(self._offset)
        

    def sample(self):
        random_state = check_random_state_instance(self._random_state)
        size = (self.n_stacks, self.n_features, self.n_features)
    
        self._W = tf.constant(random_state.normal(size = size), dtype=default_float(), name='W')
        self._S = tf.constant(chi.rvs(df=self.n_features, size= (self.n_stacks, self.n_features), random_state=random_state), dtype=default_float(), name='S')
        self._offset = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype= default_float(), name='b')
        self._latent = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=default_float(), name='latent')
        
        
    def _compute_G(self):
        G = tf.constant(np.empty([0, self.n_features]), dtype= default_float(), name='G')
        
        for i in range(self.n_stacks):
             Q, _ = tf.linalg.qr(self._W[i,:,:])
             SQ = Q @ tf.linalg.diag(self._S[i])
             G = tf.experimental.numpy.vstack([G, SQ])

        self._G = tf.transpose(G)

    
    def fit(self, sample = True):
        
        if sample : self.sample()
        
        self._compute_G()
        gamma = 1 / ( self.lengthscales **2 )

        if len(gamma.shape) == 0 or gamma.shape[0] == 1 :
            self._random_weights =  tf.math.sqrt( gamma) * self._G
            self._is_fitted = True
            return self

        self._random_weights =  tf.linalg.diag(tf.math.sqrt(gamma))  @ self._G
        self._is_fitted = True
        return self
    



class LowRankRFFOrthogonalnoOffset(LowRankRFFnoOffset):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, n_features = 1, random_state = None):
       
        super().__init__(kernel, beta0, space, n_components, n_features, 
                          random_state)
        
        self.n_stacks = int(np.ceil(self.n_components/self.n_features))
        self.n_components = self.n_stacks * self.n_features
        
        
     
    def set_points_trainable(self, trainable):
        
        if not self._is_fitted :
            raise ValueError("Random Fourrier object not fitted")

        if trainable is True :
            self._points_trainable = True
            self._W = tf.Variable(self._W)
        else :
            self._points_trainable = False
            self._W = tf.constant(self._W)
        
        

    def sample(self, latent_only = False):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (2 * self.n_components, 1)), dtype=default_float(), name='latent')
        if latent_only : return
 
        size = (self.n_stacks, self.n_features, self.n_features)
        self._W = tf.constant(random_state.normal(size = size), dtype=default_float(), name='W')
        self._S = tf.constant(chi.rvs(df=self.n_features, size= (self.n_stacks, self.n_features), random_state=random_state), dtype=default_float(), name='S')

        
    def _compute_G(self):
        G = tf.constant(np.empty([0, self.n_features]), dtype= default_float(), name='G')
        
        for i in range(self.n_stacks):
              Q, _ = tf.linalg.qr(self._W[i,:,:])
              SQ = Q @ tf.linalg.diag(self._S[i])
              G = tf.experimental.numpy.vstack([G, SQ])

        self._G = tf.transpose(G)

    
    def fit(self, sample = True):
        
        if sample : self.sample()
        
        self._compute_G()
        gamma = 1 / ( self.lengthscales **2 )

        if len(gamma.shape) == 0 or gamma.shape[0] == 1 :
            self._random_weights =  tf.math.sqrt( gamma) * self._G
            self._is_fitted = True
            return self

        self._random_weights =  tf.linalg.diag(tf.math.sqrt(gamma))  @ self._G
        self._is_fitted = True
        return self
        
        
        
    
if __name__ == "__main__": 
    import gpflow.kernels as gfk
    variance = 2
    lengthscales = 0.5
    kernel = gfk.SquaredExponential(variance= variance, lengthscales= lengthscales)
    l =  LowRankRFFOrthogonalnoOffset(kernel, space = Space(), n_components = 5, n_features = 2)  
    l.fit()
    l.set_points_trainable(True)

    with tf.GradientTape() as tape:
        l.fit(sample = False)
        #loss = l.integral()  
        #loss = tf.reduce_sum(l._G)
        Q, _ = tf.linalg.qr(l._W[1,:,:])
        SQ = Q @ tf.linalg.diag(l._S[1])
        loss = SQ
        #loss = tf.constant(loss, dtype= default_float(), name='G')
    grads = tape.gradient(loss, l.trainable_variables)
    
    print(grads)
    
    









    