import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.helper import get_process, method
from point.point_process import CoxLowRankSpatialModel, Space
from point.low_rank_rff import LowRankRFF
from point.low_rank_nystrom import LowRankNystrom

import gpflow.kernels as gfk

directory = "D:\GitHub\point\data"


rng = np.random.RandomState(40)

space = Space(-1,1) 
variance = tf.Variable(5, dtype=float_type, name='sig')
length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='l')

method = method.NYST
p = get_process(method, n_components = 500, length_scale = length_scale, variance = variance, random_state = rng )
data = p.generate(space = space, n_warm_up = 1000, n_iter = 30, batch_size  = 1, verbose = True)

directory = "D:\GitHub\point\data"
np.save(directory + "\data_synth_points.npy", data.locs)
np.save(directory + "\data_synth_sizes.npy", data.sizes)
np.save(directory + "\data_synth_variables.npy", data.variables)
np.save(directory + "\data_synth_space.npy", data.space)
