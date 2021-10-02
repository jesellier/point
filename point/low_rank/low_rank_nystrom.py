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

import gpflow.kernels as gfk
from gpflow.config import default_float

from point.utils import check_random_state_instance
from point.low_rank.low_rank_base import LowRankBase
from point.misc import Space


from enum import Enum


def tf_calc_Psi_matrix_SqExp(Z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z,z') = ∫ K(z,x) K(x,z') dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).
    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.
    Does not broadcast over leading dimensions.
    """
    variance = tf.cast(variance, Z.dtype)
    lengthscales = tf.cast(lengthscales, Z.dtype)

    mult = tf.cast(0.5 * np.sqrt(np.pi), Z.dtype) * lengthscales
    inv_lengthscales = 1.0 / lengthscales

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]

    z1 = tf.expand_dims(Z, 1)
    z2 = tf.expand_dims(Z, 0)

    zm = (z1 + z2) / 2.0

    exp_arg = tf.reduce_sum(-tf.square(z1 - z2) / (4.0 * tf.square(lengthscales)), axis=2)

    erf_val = tf.math.erf((zm - Tmin) * inv_lengthscales) - tf.math.erf(
        (zm - Tmax) * inv_lengthscales
    )
    product = tf.reduce_prod(mult * erf_val, axis=2)
    out = tf.square(variance) * tf.exp(exp_arg + tf.math.log(product))
    return out



def tf_calc_Psi_vector_SqExp(z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z) = ∫ K(z,x) dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).
    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.
    Does not broadcast over leading dimensions.
    """
    
    variance = tf.cast(variance, z.dtype)
    lengthscales = tf.cast(lengthscales, z.dtype)

    mult = tf.cast(np.sqrt(0.5 * np.pi), z.dtype) * lengthscales
    inv_lengthscales = 1.0 / lengthscales

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]
    
    erf_val = tf.math.erf(np.sqrt(0.5) * (z - Tmin) * inv_lengthscales) - tf.math.erf(np.sqrt(0.5) * (z - Tmax) * inv_lengthscales)
    product = tf.reduce_prod(mult * erf_val, axis=1)
    out =  variance * tf.expand_dims(product, 1)
    return out
    



class LowRankNystrom(LowRankBase):
    
    class mode(Enum):
        SAMPLING_SPACE = 1
        SAMPLING_DATA = 2
        GRID = 3


    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 250, X = None, random_state = None, mode = 'grid'):
        
        super().__init__(kernel, beta0, space, n_components, random_state)
        
        if not isinstance(kernel, gfk.SquaredExponential):
            raise NotImplementedError(" 'kernel' must of 'gfk.SquaredExponential' type")

        self._jitter = 1e-5
        self._do_truncation = False
        self._trunc_threshold = 1e-05
        self._X = X

        if mode == 'sampling' :
            self.mode = LowRankNystrom.mode.SAMPLING_SPACE
        elif  mode == 'grid' :
            self.mode = LowRankNystrom.mode.GRID
        elif mode == 'data_based' :
            self.mode = LowRankNystrom.mode.SAMPLING_DATA
        else :
            raise ValueError("Mode not recognized")
      
        
    def fit(self, sample = True):
        if sample : self.sample()
        self.__evd()
        self._is_fitted = True
        
        return self

    def sample(self, latent_only = False):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=default_float(), name='w')
        if latent_only : return
        
        if self.mode == LowRankNystrom.mode.SAMPLING_SPACE :
            self.__sample_x()
        elif self.mode == LowRankNystrom.mode.GRID :
            self.__grid_x()
        elif self.mode == LowRankNystrom.mode.SAMPLING_DATA :
            self.__data_x()
        else :
            raise ValueError("Mode not recognized")
        pass
  
    
    def set_data(self, X):
        self._X = X
        pass
    
    def set_truncation(self, trunc_threshold):
        self._do_truncation = True
        self._trunc_threshold = trunc_threshold
        

    def __sample_x(self):
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bounds
        sample = tf.constant(random_state.uniform(bounds[:, 0], bounds[:, 1], size=(self.n_components, self.n_features)), 
                             dtype=default_float(), 
                             name='x')
        self._x = sample
        
        
    def __data_x(self):
        if self._X is None :
            raise ValueError("No dataset instanciated")

        random_state = check_random_state_instance(self._random_state)
        shuffled_idx = np.arange(self._X.shape[0])
        random_state.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0: self.n_components]
        self._x = tf.convert_to_tensor(self._X[shuffled_idx, :], dtype=default_float())
 
    
    def __grid_x(self):
        
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bound1D
        
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
        
        sample = tf.constant(inputs, dtype=default_float(), name='x')
        self._x = sample
        

    def __evd(self):
        K = self.kernel(self._x, self._x)
        #K += self._jitter * tf.eye(K.shape[0], dtype=default_float()) 
        self._lambda, self._U, _ = tf.linalg.svd(K)

        if self._do_truncation  :
            num_truncated = tf.reduce_sum(tf.cast(self._lambda < self._trunc_threshold, tf.int64)).numpy()
            if  num_truncated  > 0 :
                n = self.n_components - num_truncated
                self._lambda = self._lambda[0:n]
                self._U = self._U[:, 0: n]
                self.n_components = self._lambda.shape[0]
                self.sample(latent_only = True)
                print("n_components recasted_to :=" + str(self.n_components))   
        pass


    def inv(self):
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        return self._U @ tf.linalg.diag(1/self._lambda) @ tf.transpose(self._U)
    
    
    def feature(self, X):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
        
        X = self.validate_entry(X)
        return self.kernel(X, self._x) @ self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda))
    
    

    def __call__(self, X):
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        X = self.validate_entry(X)
        K = self.kernel(X, self._x)
        return K @ self.inv() @ tf.transpose(K)


    def integral(self, bound = None):
        
        if bound is None :
            bound = self.space.bound
        
        variance = self.kernel.variance
        lengthscales = self.kernel.lengthscales
        
        v =  self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda)) @ self._latent
        Psi = tf_calc_Psi_matrix_SqExp(self._x, variance, lengthscales,  domain = bound )
        integral = tf.transpose(v) @ Psi @ v
        
        if self.hasOffset is True :
            m = tf_calc_Psi_vector_SqExp(self._x, variance, lengthscales,  domain = bound )
            integral += 2 * self.beta0 *  tf.transpose(m) @ v
            integral += self.beta0**2  * self.space.measure
        
        integral = integral[0][0]
        return integral
    
    
    def M(self, bound = None):
        
        if bound is None :
            bound = self.space.bound

        variance = self.kernel.variance
        lengthscales = self.kernel.lengthscales
        u  =  self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda))
        Psi = tf_calc_Psi_matrix_SqExp(self._x, variance, lengthscales,  domain = bound )
        
        return tf.transpose(u) @ Psi @ u



if __name__ == '__main__':

    import gpflow
    rng  = np.random.RandomState(10)
    X = tf.constant(rng.normal(size = [10, 2]), dtype=default_float())
    variance = tf.Variable(5, dtype=default_float(), name='sig')
    length_scale = tf.Variable(0.5, dtype=default_float(), name='lenght_scale')
    kernel = gpflow.kernels.SquaredExponential(lengthscales= 0.5, variance= 1)
    K1 = kernel(X).numpy()

    lrk = LowRankNystrom(kernel, n_components = 75, X = X, random_state= rng)
    lrk.fit()
    K2 = lrk(X).numpy()



    


    
