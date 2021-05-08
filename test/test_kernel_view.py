# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64
rng = np.random.RandomState(40)


from point.point_process import Space, CoxLowRankSpatialModel
from point.low_rank_rff import LowRankRFF
from point.low_rank_nystrom import LowRankNystrom

import unittest

import gpflow.kernels as gfk



        
class Test_Kernel(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState()
        variance = tf.Variable(2, dtype=float_type, name='sig')
        length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='l')
        period = (1000, 2) #number of periods lenght parameter per dimension (must be set)
        
        self.X =tf.constant(rng.normal(size = [500, 2]), dtype=float_type, name='X')

        kernel1 = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
        #kernel2 =  gfk.Matern12(variance= variance, lengthscales= length_scale)
        
        #variance2 = tf.Variable(0.5, dtype=float_type, name='sig')
        #poly =  gfk.Polynomial(degree=2.0, variance= variance2, offset= 0.1)
        #linear = gfk.Linear(variance = variance2)

        #comp1 = gfk.SquaredExponential(variance= variance, lengthscales= length_scale) +  poly
        #comp2 = gfk.SquaredExponential(variance= variance, lengthscales= length_scale) +  linear
        comp3 = kernel1 + gfk.Periodic(kernel1, period = period)
        
        self.kernel = comp3
        self.lrgp = LowRankNystrom(self.kernel, n_components =  250, random_state= rng, mode = 'grid').fit()
        self.process = CoxLowRankSpatialModel(self.lrgp, random_state = rng)
      

    def test_different_kernel(self):
        process = self.process
        X = self.X
        print(self.kernel)

        space = Space(-1,1)
        data = process.generate(verbose = False, n_warm_up = 10000, n_iter = 2, space = space) 
        process.lrgp.plot_kernel()
        process.lrgp.plot_surface()
        data.plot_points()


if __name__ == '__main__':
    unittest.main() 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


 
    
    