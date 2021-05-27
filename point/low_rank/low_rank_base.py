# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np

from point.misc import Space
import matplotlib.pyplot as plt

import gpflow
from gpflow.utilities.ops import  square_distance
from gpflow.base import Parameter
from gpflow.utilities import positive

import abc


class LowRankBase(gpflow.models.GPModel, metaclass=abc.ABCMeta):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 250, random_state = None):
        
        super().__init__(
            kernel,
            likelihood=None,  # custom likelihood
            num_latent_gps = 1
        )

        self.n_components = n_components
        self.n_features = 2
        self.space = space
        
        if beta0 is None :
            self.beta0 = Parameter([1e-10], transform=positive(), name = "beta0")
            gpflow.set_trainable(self.beta0, False)
        else :
            self.beta0 = Parameter(beta0, transform=positive(), name = "beta0")
        
        self._is_fitted = False
        self._random_state = random_state
        

    def lambda_func(self, X):
        return (self.func(X) + self.beta0)**2
        
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
    def __call__(self, X):
         raise NotImplementedError()

    @abc.abstractmethod 
    def integral(self, bounds = None):
         raise NotImplementedError()


    def plot_surface(self, grid_size = 40):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        
        lbound = self.space.bounds1D[0]
        hbound = self.space.bounds1D[1]
        step = (hbound - lbound) / grid_size
        
        x = y = np.arange(lbound, hbound, step)
        X, Y = np.meshgrid(x, y)
    
        n = X.shape[0]**2
        inputs = np.zeros((n,2))
        inputs[:,0] = np.ravel(X)
        inputs[:,1] = np.ravel(Y)

        zs = np.array((self.func(X = inputs) + self.beta0)**2)
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
        k = self.__call__(inputs)

        ax = plt.axes()
        x = r[:,0].numpy()
        y = k[:,0].numpy()
        ax.plot(x, y)
        
        plt.xlabel("distance")
        plt.ylabel("kernel");
        plt.show()
        pass

    

    
