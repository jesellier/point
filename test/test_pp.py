# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from sklearn.gaussian_process import GaussianProcessRegressor
import unittest

from point.low_rank_gp import LowRankApproxGP
from point.point_process import Space, HomogeneousSPP, InhomogeneousSPP, CoxLowRankSGP

rng = np.random.RandomState(40)


def print_points(x1, x2 = None):
    plt.scatter(x1[:,0], x1[:,1], edgecolor='b', facecolor='none', alpha=0.5 );
    
    if x2 is not None :
        plt.scatter(x2[:,0], x2[:,1], edgecolor='r', facecolor='none', alpha=0.5 );
        
    plt.xlabel("x"); plt.ylabel("y");
    plt.show();




#class TestIntegration(unittest.TestCase):
class TestIntegration(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
        
    def test_homogeneous_pp(self): 
        
          space = Space()
          lam = 50

          p = HomogeneousSPP(lam, rng)
          x_points = p.generate(sp = space)
          
          return x_points
            
      
    def test_inhomogeneous_pp(self): 
        
        space = Space()
        fun_lambda = lambda x,y : 20*np.exp(-(x**2+ y**2)/0.5**2)
        
        p2 = InhomogeneousSPP(functor = fun_lambda)
        x_points, x_thinned = p2.generate(sp = space)
        
        return (x_points, x_thinned)
        
        
    def test_coxgp_pp(self): 
        space = Space()
    
        #kernel = 5 * RBF(length_scale=2.5)
        #noise = 1e-6
 
        #gp = GaussianProcessRegressor(
                #kernel= kernel,
                #alpha= noise,
                #normalize_y= False,
                #optimizer = None,
                #random_state= rng
                #)
    
        #link = lambda x : x**2 
    
        #p3 = CoxSGP(gp, link, rng)
        #x_points, x_thinned = p3.generate(sp = space, verbose = 1, do_clipping = True)

    

if __name__ == '__main__':
    #unittest.main()

    x1 = TestIntegration().test_homogeneous_pp()
    #print_points(x1)
    
    (x2, x2_thinned) = TestIntegration().test_inhomogeneous_pp()
    print_points(x2, x2_thinned)
    
    #TestIntegration().test_coxgp_pp()
    

    

    
     
    
    
    

    
    


        
        
        

        
        

    
    

  
    
    
