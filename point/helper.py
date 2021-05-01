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


def get_process(length_scale, variance, method =method.RFF, n_components = 250, random_state = None ):
    
    if method == method.RFF :
        lrgp = LowRankRFF(length_scale, variance, n_components =  n_components, random_state = random_state).fit()
    
    elif method == method.NYST or method == method.NYST_GRID :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, n_components =  n_components, random_state = random_state, noise = 1e-5, mode = 'grid').fit()
    
    elif method == method.NYST or method == method.NYST_SAMPLING :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, n_components =  n_components, random_state = random_state, noise = 1e-5, mode = 'sampling').fit()
    
    else : raise ValueError("enum not recognized")

    return CoxLowRankSpatialModel(lrgp, random_state = random_state)
        
   
   
    
