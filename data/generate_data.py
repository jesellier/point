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

directory = "D:\GitHub\point\data"

rng = np.random.RandomState(40)

space = Space([-1,1]) 
variance = tf.Variable([8], dtype=float_type, name='sig')
length_scale = tf.Variable([0.2], dtype=float_type, name='l')
    
variance_poly = tf.Variable([4], dtype=float_type, name='sig')
offset_poly = tf.Variable([0.02], dtype=float_type, name='sig')

method = method.COMP_POLY
process = get_process(method, space = space, n_components = 500, 
                      length_scale = length_scale, variance = variance, 
                      variance_poly = variance_poly , offset_poly = offset_poly,
                      random_state = rng )
data = process.generate(n_warm_up = 10000, n_iter = 30, batch_size  = 1000, verbose = True)

directory = "D:\GitHub\point\data"
np.save(directory + "\data_synth_points.npy", data.locs)
np.save(directory + "\data_synth_sizes.npy", data.sizes)
np.save(directory + "\data_synth_variables.npy", data.variables)
np.save(directory + "\data_synth_space.npy", data.space)