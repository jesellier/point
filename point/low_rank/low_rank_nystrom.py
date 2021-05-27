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
        SAMPLING = 1
        GRID = 2


    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 250, random_state = None, mode = 'grid'):
        
        super().__init__(kernel, beta0, space, n_components, random_state)
        
        if not isinstance(kernel, gfk.SquaredExponential):
            raise NotImplementedError(" 'kernel' must of 'gfk.SquaredExponential' type")

        self._jitter = 1e-5

        if mode == 'sampling' :
            self.mode = LowRankNystrom.mode.SAMPLING
        else :
            self.mode = LowRankNystrom.mode.GRID
      

        
    def fit(self, sample = True):
        if sample : self.sample()

        self.__evd()
        self._v =  self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda)) @ self._latent
        self._is_fitted = True
        
        return self

    def sample(self):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=default_float(), name='beta')
        
        if self.mode == LowRankNystrom.mode.SAMPLING :
            self.__sample_x()
        elif self.mode == LowRankNystrom.mode.GRID :
            self.__grid_x()
        else :
            raise ValueError("Mode not recognized")
  
        self.fit(sample = False)

   
    def __sample_x(self):
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bounds
        sample = tf.constant(random_state.uniform(bounds[:, 0], bounds[:, 1], size=(self.n_components, self.n_features)), 
                             dtype=default_float(), 
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
        
        sample = tf.constant(inputs, dtype=default_float(), name='x')
        self._x = sample
        
        
        
    def __evd(self):
        K = self.kernel(self._x, self._x)
        K_jitter_matrix = self._jitter * tf.eye(K.shape[0], dtype=default_float()) 
        K += K_jitter_matrix
        self._lambda, self._U, _ = tf.linalg.svd(K)
        
        
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
        out = self.kernel(X, self._x) @ self._U @ tf.linalg.diag(1/self._lambda)

        return out
    
    
    def func(self, X) :
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        
        X = self.__validate_entry(X)
        return self.kernel(X, self._x) @ self._v


    def __call__(self, X):
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        X = self.__validate_entry(X)
        K = self.kernel(X, self._x)
        return K @ self.inv() @ tf.transpose(K)


    def integral(self, bounds = None):
        
        if bounds is None :
            bounds = self.space.bounds
        
        variance = self.kernel.variance
        lengthscales = self.kernel.lengthscales
    
        M = tf_calc_Psi_matrix_SqExp(self._x, variance, lengthscales,  domain = bounds )
        integral = tf.transpose(self._v) @ M @ self._v
        
        m = tf_calc_Psi_vector_SqExp(self._x, variance, lengthscales,  domain = bounds )
        integral += 2 * self.beta0 *  tf.transpose(m) @ self._v
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

    

if __name__ == "__main__":

    rng  = np.random.RandomState(10)
    variance = tf.Variable(5, dtype=default_float(), name='sig')
    length_scale = tf.Variable(0.5, dtype=default_float(), name='lenght_scale')
    kernel = gfk.SquaredExponential(lengthscales= length_scale, variance= variance )
    
    lrgp = LowRankNystrom(kernel, n_components = 250, random_state=rng, mode = 'grid').fit()
    print(lrgp.integral())
    X = tf.constant(rng.normal(size = [10, 2]), dtype=default_float(), name='X')
    #print(lrgp.maximum_log_likelihood_objective(X))
    #print(lrgp.func(X))

    lrgp.plot_kernel()
    lrgp.plot_surface()

    

    
