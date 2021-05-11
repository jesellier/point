# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021
@author: jesel
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from point.point_process import CoxLowRankSpatialModel
from point.low_rank.low_rank_rff import LowRankRFF
from point.low_rank.low_rank_nystrom import LowRankNystrom
from point.misc import Space

import gpflow.kernels as gfk
from gpflow.config import default_float

from enum import Enum


def defaultArgs():
    out = dict(length_scale = tf.constant([0.5], dtype=default_float(), name='lengthscale'),
               variance = tf.constant([5], dtype=default_float(), name='variance'),
               beta0 = None
               )
    
    return out

class method(Enum):
    NYST = 1
    RFF = 2
    NYST_SAMPLING = 3
    NYST_GRID = 4
    COMP_POLY = 5
    
    
def get_lrgp(method =method.RFF, space = Space(), n_components = 250, random_state = None, **kwargs):
    
    lrgp = None
    kwargs = {**defaultArgs(), **kwargs} #merge kwards with default args (with priority to args)
    length_scale = kwargs['length_scale']
    variance = kwargs['variance']
    beta0 = kwargs['beta0']

    if method == method.RFF :
        kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
        lrgp = LowRankRFF(kernel, beta0 = beta0,space = space, n_components =  n_components, random_state = random_state)
        lrgp.fit()
    
    elif method == method.NYST or method == method.NYST_GRID :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components, random_state = random_state)
         lrgp.fit()
    
    elif method == method.NYST or method == method.NYST_SAMPLING :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components, random_state = random_state, mode = 'sampling')
         lrgp.fit()

    return lrgp



def get_process(method =method.RFF, n_components = 250, random_state = None, **kwargs):
    lrgp = get_lrgp(method =method , n_components = n_components, random_state = random_state, **kwargs)
    return CoxLowRankSpatialModel(lrgp, random_state = random_state)
        
   
   
    
