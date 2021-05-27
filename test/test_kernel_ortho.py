# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64
rng = np.random.RandomState(40)

import gpflow.kernels as gfk

from scipy.linalg import qr_multiply
from scipy.stats import chi

rng = np.random.RandomState()

        
def _get_random_matrix(distribution):
    return lambda rng, size: get_random_matrix(rng, distribution, size)

def standard_gaussian(random_state, size):
    return random_state.randn(*size)


def rademacher(random_state, size):
    return random_state.randint(2, size=size, dtype=np.int32)*2-1


def laplace(random_state, size):
    return random_state.laplace(0, 1. / np.sqrt(2), size)


def uniform(random_state, size):
    return random_state.uniform(-np.sqrt(3), np.sqrt(3), size)


def get_random_matrix(random_state, distribution, size, p_sparse=0.,
                      dtype=np.float64):
    # size = (n_components, n_features)
    if distribution == 'rademacher':
        return rademacher(random_state, size).astype(dtype)
    elif distribution in ['gaussian', 'normal']:
        return standard_gaussian(random_state, size)
    elif distribution == 'uniform':
        return uniform(random_state, size)
    elif distribution == 'laplace':
        return laplace(random_state, size)
    else:
        raise ValueError('{} distribution is not implemented. Please use'
                         'rademacher, gaussian (normal), uniform or laplace.'
                         .format(distribution))


n_features = 2
n_components = 250

variance = tf.Variable(5, dtype=float_type, name='sig')
length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')

kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
X = tf.constant(rng.normal(size = [250, 2]), dtype=float_type, name='X')
Ktrue = kernel(X).numpy()
    


# %% ######################## STANDARD RFF
random_state = rng
size = (n_features, n_components)

z = tf.constant(random_state.normal(size = size), dtype=float_type, name='z')
random_offset = tf.constant(random_state.uniform(0, 2 * np.pi, size= n_components), dtype=float_type, name='b')
beta = tf.constant(random_state.normal(size = (n_components, 1)), dtype=float_type, name='beta')

gamma = 1 / (2 * length_scale **2 )
random_weights =  tf.linalg.diag(tf.math.sqrt(2 * gamma))  @ z

feature = X @ random_weights  + random_offset
feature= tf.cos(feature)
feature= tf.sqrt(2 * variance /tf.constant(n_components, dtype=float_type)) * feature

K = feature @ tf.transpose(feature)
error = np.linalg.norm(Ktrue - K, ord = 'fro') / np.linalg.norm(Ktrue, ord = 'fro') 
print(error)


# %% ##################### ORTHO RFF
n_samples, n_features = X.shape
n_stacks = int(np.ceil(n_components/n_features))
n_components = n_stacks * n_features
    
distribution="gaussian"
random_fourier=True 
use_offset= True

size = (n_features, n_features)
if isinstance(distribution, str):
    distribution = _get_random_matrix(distribution)
else:
    distribution = distribution
random_weights = []

for _ in range(n_stacks):
    W = distribution(random_state, size)
    S = np.diag(chi.rvs(df=n_features, size=n_features,
                        random_state=random_state))
    SQ, _ = qr_multiply(W, S)
    random_weights += [SQ]

random_weights = np.vstack(random_weights).T
random_offset = None
if random_fourier:
    random_weights = np.diag(np.sqrt(2 * gamma))  @ random_weights
    if use_offset:
        random_offset = random_state.uniform(0, 2*np.pi,size=n_components)
        

feature = X.numpy() @ random_weights  + random_offset
feature= np.cos(feature)
feature= np.sqrt(2 * variance /n_components) * feature

K2 = feature @ tf.transpose(feature)
error2 = np.linalg.norm(Ktrue - K2, ord = 'fro') / np.linalg.norm(Ktrue, ord = 'fro') 
print(error2)





 

