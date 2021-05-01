# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np
import scipy as scp

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

import gpflow.kernels as gfk

from sklearn.gaussian_process.kernels import RBF

from point.utils import check_random_state_instance
from point.misc import Space
import matplotlib.pyplot as plt


def eigvalsh_to_eps(spectrum, cond=None, rcond=None):
        """
        Determine which eigenvalues are "small" given the spectrum.
        This is for compatibility across various linear algebra functions
        that should agree about whether or not a Hermitian matrix is numerically
        singular and what is its numerical matrix rank. This is designed to be compatible with scipy.linalg.pinvh.
        Parameters
        ----------
        spectrum : 1d ndarray
            Array of eigenvalues of a Hermitian matrix.
        cond, rcond : float, optional
            Cutoff for small eigenvalues.
            Singular values smaller than rcond * largest_eigenvalue are
            considered zero.
            If None or -1, suitable machine precision is used.
        Returns
        -------
        eps : float
            Magnitude cutoff for numerical negligibility.
        """
        if rcond is not None:
            cond = rcond
        if cond in [None, -1]:
            t = spectrum.dtype.char.lower()
            factor = {'f': 1E3, 'd': 1E6}
            cond = factor[t] * np.finfo(t).eps
        eps = cond * np.max(abs(spectrum))
        return eps
    


class LowRankNystrom():
    
    
    def __init__(self, kernel, n_components = 250, random_state = None, noise = 1e-5, mode = 'sampling'):
        
        self.kernel = kernel
        self.n_components = n_components
        self.n_features = 2
        self.is_fitted = False
        
        self.random_state = random_state
        self.mode = mode
        
        self._noise = noise
        
        
        
    @property
    def trainable_variables(self):
        return self.kernel.trainable_variables
    
    @property
    def parameters(self):
        return self.kernel.parameters

        
    def fit(self, sample = True):
        if sample : self.sample()

        self.__evd()
        self._vl = self._v @ self.latent_
        self.is_fitted = True
        
        return self

    def sample(self, sp = Space()):
        random_state = check_random_state_instance(self.random_state)
        self.latent_ = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=float_type, name='beta')
        
        if self.mode == "sampling" :
            self.__sample_x(sp)
        elif self.mode == "grid" :
            self.__grid_x(sp)
        else :
            raise ValueError("Mode not recognized")
            
        self.fit(sample = False)

   
    def __sample_x(self, sp):
        random_state = check_random_state_instance(self.random_state)
        bounds = sp.bounds
        sample = tf.constant(random_state.uniform(bounds[:, 0], bounds[:, 1], size=(self.n_components, self.n_features)), 
                             dtype=float_type, 
                             name='x')
        
        self._x = sample
        
        
    def __grid_x(self, sp):
        bounds = sp.bounds
        step = 1/np.sqrt(self.n_components )
        x = np.arange(bounds[0,0], bounds[0,1], step)
        y = np.arange(bounds[1,0], bounds[1,1], step)
        X, Y = np.meshgrid(x, y)
    
        n = X.shape[0]**2
        inputs = np.zeros((n,2))
        inputs[:,0] = np.ravel(X)
        inputs[:,1] = np.ravel(Y)
        sample = tf.constant(inputs, dtype=float_type, name='x')
        self._x = sample
        
        
    def __evd(self):
        K = self.kernel(self._x, self._x)
        K = K + tf.eye(K.shape[0], dtype=float_type) * tf.constant(self._noise, dtype=float_type) 
        self._lambda, U, V = tf.linalg.svd(K)
        self._v  = U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda)) 

        #eps = eigvalsh_to_eps(s, None, None)
        #if np.min(s) < -eps:
            #raise ValueError('the input matrix must be positive semidefinite')
        #d = s[s > eps]
        #if len(d) < len(s) and not self._allow_singular:
            #raise np.linalg.LinAlgError('singular matrix')
            
      
    def inv(self):
        if not self.is_fitted :
            raise ValueError("instance not fitted")
        return self._v @ tf.transpose(self._v)
    
    
    def func(self, X) :
        if not self.is_fitted :
            raise ValueError("instance not fitted")
        return self.kernel(X, self._x) @ self._vl

    
    def kernel(self, X):
        if not self.is_fitted :
            raise ValueError("instance not fitted")
        K = self.kernel(X, self._x)
        return K @ self.inv() @ tf.transpose(K)
        

    
    def integral(self, lplus = 1.0, lminus = 0):
        out = tf.transpose(self.latent_) @ tf.linalg.diag(1/tf.math.sqrt(self._lambda)) @ self.latent_
        return out
        

    
    def likelihood(self, X, lplus = 1.0, lminus = 0):
        
        int_term = self.integral(lplus, lminus)

        f = self.func(X)
        sum_term = tf.norm(f)
        sum_term = tf.math.square(sum_term)
            
        out = - int_term + sum_term
        
        return out
    
       
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

    
    
    
if __name__ == "__main__":

    rng  = np.random.RandomState()
    variance = tf.Variable(0.5, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='lenght_scale')
    
    # noise = 1e-5
    n_components = 250
    
    # sp = Space()
    X = tf.constant(rng.normal(size = [500, 2]), dtype=float_type, name='X')
    kernel = gfk.SquaredExponential(variance= variance , lengthscales= length_scale)
    nyst = LowRankNystrom(kernel, n_components , random_state=rng).fit()
    nyst.plot_surface()
    nyst.sample()

    x_tries = rng.uniform([-1,-1], [1,1] , size=(10000, 2))
    fs = nyst.func(tf.constant(x_tries, dtype=float_type))**2
    fs = fs.numpy()
    
    # noise = 1e-5
    # n_components = 250
    
    # sp = Space()
    X = tf.constant(rng.normal(size = [500, 2]), dtype=float_type, name='X')
    


    # K1 = kernel(X,X).numpy()
    
    # nyst = LowRankNystrom(kernel, n_components , random_state=rng)
    # nyst.sample()
    # #nyst.fit()
    
    # x = nyst._LowRankNystrom__x
    # Kxx = kernel(x, x)
    # Kxx = Kxx + tf.eye(Kxx.shape[0], dtype=float_type) * tf.constant(noise, dtype=float_type) 
    # s, U, V = tf.linalg.svd(Kxx)
    # v = U @ tf.linalg.diag(1/tf.math.sqrt(s)) 

    # beta = tf.constant(rng.normal(size = (n_components, 1)), dtype=float_type, name='beta')
    # vl = v @ beta
    
    # Kx = kernel(X, x)
    # func = Kx @ vl

    # inv = v @ tf.transpose(v)
    
    # K = Kx @ inv @ tf.transpose(Kx)
    
    # out = (inv @ Kxx) #+ 2 * (kernel.trainable_parameters[1])
   
    
    
    
    
    # #######################
    # # compare with sklearn
    
    # kernel2 =  0.5 * RBF([0.5, 100])
    # K2 = kernel2(X)
    
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
    
    

    

    
