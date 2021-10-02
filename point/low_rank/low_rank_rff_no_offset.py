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

def expamdedSumPerDimension(x):
    Mp, Mm = expandedSum(x)
    
    d1 = tf.stack([Mp[:,:,0] , tf.linalg.set_diag(Mm[:,:,0], tf.ones(x.shape[0],  dtype=default_float())) ])
    d2 = tf.stack([Mp[:,:,1] , tf.linalg.set_diag(Mm[:,:,1], tf.ones(x.shape[0],  dtype=default_float())) ])

    return (d1, d2)



class LowRankRFFnoOffset(LowRankBase):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, random_state = None):
       
        super().__init__(kernel, beta0, space, n_components, random_state)
        
        if not isinstance(kernel, gfk.SquaredExponential):
            raise NotImplementedError(" 'kernel' must of 'gfk.SquaredExponential' type")


    def sample(self, latent_only = False):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (2 * self.n_components, 1)), dtype=default_float(), name='latent')
        if latent_only : return
        
        size = (self.n_features, self.n_components)
        self._G = tf.constant(random_state.normal(size = size), dtype=default_float(), name='G')
        pass


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
    
    
    def feature(self, X, get_grad = False):
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
 
        prod = X @ self._random_weights
        features = tf.experimental.numpy.hstack([tf.cos(prod),tf.sin(prod)])
        features *= tf.sqrt(self.variance / self.n_components)
        
        if self.lengthscales.shape == [] :
                l1 = self.lengthscales
                l2 = self.lengthscales
        else :
                l1 = self.lengthscales[0]
                l2 = self.lengthscales[1]
        
        if get_grad is True :
            dv = 0.5 * features / self.variance
            
            v1 = tf.expand_dims(X[:,0],1) * tf.transpose(tf.expand_dims(self._random_weights[0,:],1))
            dl1 = tf.experimental.numpy.hstack([ features[: , self.n_components :]  * v1, - features[:, 0: self.n_components]  * v1]) / l1
            
            v2 = tf.expand_dims(X[:,1],1) * tf.transpose(tf.expand_dims(self._random_weights[1,:],1))
            dl2 = tf.experimental.numpy.hstack([features[: , self.n_components :]  * v2, - features[:, 0: self.n_components]  * v2]) / l2
            
            grads = tf.experimental.numpy.vstack([
                tf.expand_dims(dl1,0),
                tf.expand_dims(dl2,0),
                tf.expand_dims(dv,0)
                ])
            return (features, grads)
            

        return features
    

    def __call__(self, X, X2 = None):
        if X2 is None :
            Z = self.feature(X)
            return Z @ tf.transpose(Z)
        return  self.feature(X)  @ tf.transpose(self.feature(X2))
 


    def M(self, bound = None, get_grad = False):
        
        if bound is None :
            bound = self.space.bound1D
            
        R =  self.n_components
        
        B, A  = self.__M(bound, get_grad)
        cache = tf.linalg.diag(tf.ones(R,  dtype=default_float()))
        zeros = tf.zeros((R,R),  dtype=default_float())
        cache1 = tf.experimental.numpy.hstack([cache , zeros])
        cache2 = tf.experimental.numpy.hstack([zeros, cache]) 
        
        M =  tf.transpose(cache1) @ (A + B) @ cache1 + tf.transpose(cache2) @ (A - B) @ cache2

        if get_grad :
            out, dl1, dl2 = M
            dv = tf.expand_dims(out /self.variance,0)
            return (out, tf.experimental.numpy.vstack([ tf.expand_dims(dl1,0), tf.expand_dims(dl2,0), dv]))
        
        return M


    def integral(self, bound = None, get_grad = False):
        
        if bound is None :
            bound = self.space.bound1D
            
        B, A = self.__M(bound, get_grad )
        R =  self.n_components
        
        cache = tf.linalg.diag(tf.ones(R,  dtype=default_float()))
        zeros = tf.zeros((R,R),  dtype=default_float())

        w1 = tf.experimental.numpy.hstack([cache , zeros]) @ self._latent
        w2 = tf.experimental.numpy.hstack([zeros, cache]) @ self._latent
        integral = tf.transpose(w1) @ (A + B) @ w1 + tf.transpose(w2) @ (A - B) @ w2
        
        add_to_out = 0.0
        sub_to_out = 0.0
        
        if self.hasOffset is True :
            beta_term = 2 * self.beta0 *  tf.transpose(w1) @ self.__m(bound, get_grad)
            integral += beta_term 
            add_to_out = self.beta0**2 *self.space.measure
            sub_to_out = beta_term[0][0] 
            
        if get_grad :
            out, dl1, dl2 = integral
            dv = (out - 0.5 * sub_to_out) /self.variance   # dtotal/dv = quad_term/variance + 0.5 beta_term/variance. 
            out += add_to_out
            grads = tf.experimental.numpy.vstack([dl1, dl2, dv])
            return (out[0][0], grads)

        return integral[0][0] + add_to_out

    
    def __M(self, bound = [-1,1], get_grad = False):
        # Return the matrices B and A
        # without grad return : M = [B,A] (i.e. 2xRxR tensor)
        # with grad return : M = [[B, der1B, der2B], [A, der1A, der2A]] (i.e. 2x3xRxR tensor)
        
        
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        R =  self.n_components 
        z1 = self._random_weights[0,:]
        z2 = self._random_weights[1,:]

        d1, d2 =  expamdedSumPerDimension(tf.transpose(self._random_weights))

        sin_d1 = tf.sin(bound*d1) 
        sin_d2 = tf.sin(bound*d2)

        M = (2 / (d1 * d2)) * sin_d1  * sin_d2
        diag = tf.stack([(1 / (2 * z1* z2)) * tf.sin(2*bound*z1) * tf.sin(2 *bound*z2), 2 * bound ** 2 * tf.ones(R,  dtype=default_float())])                                     
        M = tf.linalg.set_diag(M, diag) 

        if get_grad :

            if self.lengthscales.shape == [] :
                l1 = self.lengthscales
                l2 = self.lengthscales
            else :
                l1 = self.lengthscales[0]
                l2 = self.lengthscales[1]

            dl1 = - ( 2 * bound * tf.cos(bound*d1) * sin_d2 / d2 - M ) / l1
            dl1 =  tf.linalg.set_diag(dl1, tf.stack([ - ( bound * tf.cos(2*bound*z1) * tf.sin(2*bound*z2) / z2 - diag[0,:] ) / l1, tf.zeros(R,  dtype=default_float())]) ) 

            dl2 = - ( 2 * bound * sin_d1 * tf.cos(bound*d2) / d1 - M ) / l2
            dl2 =  tf.linalg.set_diag(dl2, tf.stack([ - ( bound * tf.sin(2*bound*z1) * tf.cos(2*bound*z2) / z1 - diag[0,:] ) / l2, tf.zeros(R,  dtype=default_float())])) 

            out = tf.experimental.numpy.vstack([
                tf.expand_dims( tf.experimental.numpy.vstack([tf.expand_dims(M[0,:],0), tf.expand_dims(dl1[0,:],0), tf.expand_dims(dl2[0,:],0)]),0),
                tf.expand_dims(tf.experimental.numpy.vstack([tf.expand_dims(M[1,:],0), tf.expand_dims(dl1[1,:],0), tf.expand_dims(dl2[1,:],0)]),0)
                ])

            return self.variance * out / R
            
        return self.variance * M / R
 
    
    
    def __m(self, bound = [-1,1], get_grad = False):
 
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]

        R =  self.n_components
        z1 = self._random_weights[0,:]
        z2 = self._random_weights[1,:]
        
        sin_z1 = tf.sin(bound*z1) 
        sin_z2 = tf.sin(bound*z2) 

        vec = 4* sin_z1 * sin_z2 
        vec =  tf.linalg.diag(1 / (z1 * z2)) @ tf.expand_dims(vec, 1)
        factor = tf.sqrt(tf.convert_to_tensor( self.variance/ R, dtype=default_float()))

        if get_grad is True :
            
            if self.lengthscales.shape == [] :
                l1 = self.lengthscales
                l2 = self.lengthscales
            else :
                l1 = self.lengthscales[0]
                l2 = self.lengthscales[1]
                
            dl1 =  - ( np.expand_dims(4 * bound * tf.cos(bound*z1) * sin_z2 / z2,1) - vec ) / l1
            dl2 =  - ( np.expand_dims(4 * bound * sin_z1 * tf.cos(bound*z2) / z1,1) - vec ) / l2
            return factor * tf.experimental.numpy.vstack([tf.expand_dims(vec,0), tf.expand_dims(dl1,0), tf.expand_dims(dl2,0)])
 
        return  factor * vec

      
