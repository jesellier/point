
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from point.model import CoxLowRankSpatialModel
from point.low_rank.low_rank_rff_with_offset import LowRankRFF
from point.low_rank.low_rank_rff_ortho import LowRankRFFOrthogonal, LowRankRFFOrthogonalnoOffset
from point.low_rank.low_rank_rff_no_offset import LowRankRFFnoOffset
from point.low_rank.low_rank_nystrom import LowRankNystrom
from point.misc import Space
from point.laplace import LaplaceApproximation

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
    RFF_WITH_OFFSET = 3
    RFF_NO_OFFSET = 4
    RFF_ORTHO = 5
    RFF_ORTHO_NO_OFFSET = 6
    NYST_DATA = 7
    NYST_GRID = 8
    NYST_SAMPLING  = 9

    
def get_lrgp(method = method.RFF, space = Space(), n_components = 250, n_features = 2, random_state = None, **kwargs):
    
    lrgp = None
    kwargs = {**defaultArgs(), **kwargs} #merge kwards with default args (with priority to args)
    length_scale = kwargs['length_scale']
    variance = kwargs['variance']
    beta0 = kwargs['beta0']

    if method == method.RFF or method == method.RFF_WITH_OFFSET  :
        kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
        lrgp = LowRankRFF(kernel, beta0 = beta0, space = space, n_components =  n_components, n_features = n_features, random_state = random_state)
        lrgp.fit()
        
    elif method == method.RFF_ORTHO   :
        kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
        lrgp = LowRankRFFOrthogonal(kernel, beta0 = beta0, space = space, n_components =  n_components, n_features = n_features, random_state = random_state)
        lrgp.fit()

    elif method == method.RFF_NO_OFFSET:
        kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
        lrgp = LowRankRFFnoOffset(kernel, beta0 = beta0, space = space, n_components =  n_components, n_features = n_features, random_state = random_state)
        lrgp.fit()
        
    elif method ==  method.RFF_ORTHO_NO_OFFSET :
        kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
        lrgp = LowRankRFFOrthogonalnoOffset(kernel, beta0 = beta0, space = space, n_components =  n_components, n_features = n_features, random_state = random_state)
        lrgp.fit()
        
        
    elif method == method.NYST or method == method.NYST_DATA :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components, n_features = n_features, random_state = random_state, sampling_mode = 'data_based')

    elif method == method.NYST_GRID :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components,  n_features = n_features, random_state = random_state, sampling_mode = 'grid')
         lrgp.fit()
    
    elif method == method.NYST_SAMPLING :
         kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components, n_features = n_features, random_state = random_state, sampling_mode = 'sampling')
    else :
         raise ValueError('method type not recognized')

    return lrgp



def get_process(name = None, method = method.RFF, n_components = 250, n_features = 2, random_state = None, **kwargs):
    lrgp = get_lrgp(method =method, n_components = n_components, n_features = n_features, random_state = random_state, **kwargs)
    return CoxLowRankSpatialModel(lrgp, name, random_state = random_state)
        
   
    
def get_rff_model(name = "model", n_dims = 2, n_components = 75, method = method.RFF_NO_OFFSET, variance = 2.0, space = Space(), random_state = None):
    
    name = name + ".rff." + str(n_components)
    variance = tf.Variable(variance, dtype=default_float(), name='sig')
    length_scale = tf.Variable(n_dims  * [0.5], dtype=default_float(), name='lenght_scale')

    model = get_process(
        name = name,
        length_scale = length_scale, 
        variance = variance, 
        method = method,
        space = space,
        n_components = n_components, 
        n_features = n_dims, 
        random_state = random_state)
    
    lp = LaplaceApproximation(model) 

    return lp


def get_nyst_model(name = "nyst", n_dims = 2, n_components = 75, variance = 2.0, space = Space(), random_state = None):
    
    if name is None : 
        name = "nyst." + str(n_components)
    else :
        name = name + "." + str(n_components)
        
    variance = tf.Variable(variance, dtype=default_float(), name='sig')
    length_scale = tf.Variable(n_dims  * [0.5], dtype=default_float(), name='lenght_scale')
    
    model = get_process(
        name = name,
        length_scale = length_scale, 
        variance = variance, 
        method =  method.NYST_DATA,
        space = space,
        n_components = n_components, 
        n_features = n_dims, 
        random_state = random_state)
    
    lp = LaplaceApproximation(model) 
    
    return lp
