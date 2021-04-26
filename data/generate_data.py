import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.point_process import CoxLowRankSpatialModel, Space

directory = "D:\GitHub\point\data"


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng = np.random.RandomState(40)

variance = tf.Variable(5.0, dtype=float_type, name='sig')

length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='l')
       
space = Space(-1,1) 
p = CoxLowRankSpatialModel(length_scale=length_scale, variance = variance, n_components = 500, random_state = rng)
data = p.generate(sp = space, batch_size  = 100, verbose = True)

directory = "D:\GitHub\point\data"
np.save(directory + "\data_synth_points.npy", data._locs)
np.save(directory + "\data_synth_sizes.npy", data._sizes)
np.save(directory + "\data_synth_variables.npy", data._variables)
np.save(directory + "\data_synth_space.npy", data._space)
