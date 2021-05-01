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

import unittest
import time

from point.point_process import CoxLowRankSpatialModel, Space
from scipy.optimize import minimize #For optimizing


        
        
class Check_Optimization():
    
    
    #def setUp(self):
    def __init__(self):
        rng = np.random.mtrand._rand
        self.variance = tf.Variable(8, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        self.process = CoxLowRankSpatialModel(length_scale=self.length_scale, 
                                              variance = self.variance, 
                                              n_components = 500, 
                                              random_state = rng)
        self.process.fit()

    def run(self):
        process = self.process
        sp = Space(-1,1)
        bounds = sp.bounds
  
        #SET1
        t0 = time.time()
        out = process._CoxLowRankSpatialModel__optimizeBound(n_warm_up = 1000, n_iter = 20,sp = sp)[0]
        print("SET1 := %f - in [%f] " % (out, time.time() - t0))
        
        #SET2
        t0 = time.time()
        out = process._CoxLowRankSpatialModel__optimizeBound(n_warm_up = 10000, n_iter = 3,sp = sp)[0]
        print("SET2 := %f - in [%f] " % (out,time.time() - t0))
        
        #SET3
        t0 = time.time()
        out = process._CoxLowRankSpatialModel__optimizeBound(n_warm_up = 10000, n_iter = 0,sp = sp)[0]
        print("SET3 := %f - in [%f] " % (out,time.time() - t0))
        
        #SET1
        t0 = time.time()
        func = lambda x: - (process._CoxLowRankSpatialModel__func(x)**2)
        res = minimize(func, sp.center, bounds=sp.bounds)
        print("SET4 := %f - in [%f] " % (-res.fun, time.time() - t0))
        
        #BENCH
        t0 = time.time()
        x_tries = rng.uniform(bounds[:, 0], bounds[:, 1], size=(10000, 2))
        fs = process.lrgp_.func(tf.constant(x_tries, dtype=float_type))**2
        fs = fs.numpy()
        print("BENCH := %f - in [%f] " % (fs.max(), time.time() - t0))



if __name__ == '__main__':
    #unittest.main()
    Check_Optimization().run()

 
    
    