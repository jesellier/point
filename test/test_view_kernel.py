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
        variance = tf.Variable(1, dtype=float_type, name='sig')
        length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='l')

        #self.kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
        self.kernel =  gfk.Matern12(variance= variance, lengthscales= length_scale) +  gfk.Polynomial(degree=2.0, variance= variance, offset= 1.0)
        #self.kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale) +  gfk.Polynomial(degree=2.0, variance=1.0, offset= 1.0)
        
        self.lrgp = LowRankNystrom(self.kernel, n_components =  250, random_state = rng, mode = 'grid').fit()
        self.process = CoxLowRankSpatialModel(self.lrgp, random_state = rng)


    def test_different_kernel(self):
        process = self.process
        print(self.kernel)

        space = Space(-1,1)
        data = process.generate(verbose = False, n_warm_up = 10000, n_iter = 2, space = space) 
        data.plot_points()
        process.lrgp.plot_surface()



if __name__ == '__main__':
    unittest.main()


 
    
    