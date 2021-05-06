# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.point_process import CoxLowRankSpatialModel
from point.low_rank_rff import LowRankRFF
from point.low_rank_nystrom import LowRankNystrom

import gpflow.kernels as gfk

from enum import Enum


class method(Enum):
    NYST = 1
    RFF = 2
    NYST_SAMPLING = 3
    NYST_GRID = 4
    COMP_POLY = 5
    
    
def get_lrgp(method =method.RFF, n_components = 250, random_state = None, **kwargs):
    
    lrgp = None
    
    
    length_scale = kwargs['length_scale']
    variance = kwargs['variance']
    
    if method == method.RFF :
        lrgp = LowRankRFF(length_scale, variance, n_components =  n_components, random_state = random_state)
        lrgp.fit()
    
    elif method == method.NYST or method == method.NYST_GRID :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, n_components =  n_components, random_state = random_state)
         lrgp.fit()
    
    elif method == method.NYST or method == method.NYST_SAMPLING :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, n_components =  n_components, random_state = random_state, mode = 'sampling')
         lrgp.fit()

    elif method == method.COMP_POLY :
         
        variance2 = kwargs['variance_poly']
        offset = kwargs['offset_poly']
        
        poly =  gfk.Polynomial(degree=2.0, variance= variance2, offset= offset)
        comp = gfk.SquaredExponential(variance= variance, lengthscales= length_scale) +  poly
        lrgp = LowRankNystrom(kernel = comp, n_components =  n_components, random_state = random_state)
        lrgp.fit()
        
    return lrgp



def get_process(method =method.RFF, n_components = 250, random_state = None, **kwargs):
    
    lrgp = get_lrgp(method =method , n_components = n_components, random_state = random_state, **kwargs)

    return CoxLowRankSpatialModel(lrgp, random_state = random_state)
        
   
   
    
