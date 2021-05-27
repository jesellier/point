# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:41:15 2021

@author: jesel
"""


import numpy as np

import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.helper import get_process, method
from point.point_process import Space


rng = np.random.RandomState(40)
space = Space([-10,10]) 
variance = tf.Variable([5], dtype=float_type, name='sig')
length_scale = tf.Variable([0.5], dtype=float_type, name='lengthscale')
beta0 = tf.Variable([0.2], dtype=float_type, name='lengthscale')

method = method.RFF
process = get_process(method, space = space, n_components = 500, 
                      length_scale = length_scale, variance = variance, beta0 = beta0,
                      random_state = rng )
data = process.generate(n_warm_up = 10000, n_iter = 30, batch_size  = 1000, verbose = True)

directory = "D:\GitHub\point\data"
np.save(directory + "\data_synth_points.npy", data.locs)
np.save(directory + "\data_synth_sizes.npy", data.sizes)
np.save(directory + "\data_synth_variables.npy", data.variables)
np.save(directory + "\data_synth_space.npy", data.space)