# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.misc import Space
import matplotlib.pyplot as plt

from gpflow.utilities.ops import  square_distance

import abc



class LowRankBase(metaclass=abc.ABCMeta):
    

    def __init__(self, space = Space(), n_components = 250, random_state = None):

        self.n_components = n_components
        self.n_features = 2
        self.space = space
        
        self._is_fitted = False
        self._random_state = random_state

        

    @property
    @abc.abstractmethod
    def trainable_variables(self):
        return self._impl_kernel.trainable_variables
    
    @property
    @abc.abstractmethod
    def parameters(self):
        return self._impl_kernel.parameters

    @abc.abstractmethod 
    def fit(self, sample = True):
        raise NotImplementedError()
        
    @abc.abstractmethod 
    def sample(self):
        raise NotImplementedError()
        
    @abc.abstractmethod 
    def feature(self, X):
        raise NotImplementedError()

    @abc.abstractmethod 
    def func(self, X) :
          raise NotImplementedError()
          
    @abc.abstractmethod 
    def kernel(self, X):
         raise NotImplementedError()

    @abc.abstractmethod 
    def integral(self, bounds = None):
         raise NotImplementedError()

        
    @abc.abstractmethod 
    def likelihood(self, X, bounds = None):
         raise NotImplementedError()

    
       
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
        ax.set_zlabel('Intensity')
        plt.show()
        pass
    
    
    def plot_kernel(self):
        plt.figure()
        x = np.arange(-1.0, 1.0, 0.0255)
        
        n = len(x)
        origin = inputs = np.zeros((n,2))
        inputs[:,1] = x

        r = square_distance(origin, inputs)
        k = self.kernel(inputs)

        ax = plt.axes()
        x = r[:,0].numpy()
        y = k[:,0].numpy()
        ax.plot(x, y)
        
        plt.xlabel("distance")
        plt.ylabel("kernel");
        plt.show()
        pass

    

    
