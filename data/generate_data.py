import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.point_process import CoxLowRankSpatialModel, Space
from point.low_rank_rff import LowRankRFF
from point.low_rank_nystrom import LowRankNystrom

directory = "D:\GitHub\point\data"


rng = np.random.RandomState(40)

space = Space(-1,1) 
variance = tf.Variable(5.0, dtype=float_type, name='sig')
length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='l')

lrgp = LowRankRFF(length_scale , variance,  n_components = 500, random_state = rng).fit()
p = CoxLowRankSpatialModel(lrgp, random_state = rng)
data = p.generate(sp = space, n_warm_up = 10000, n_iter = 30, batch_size  = 1000, verbose = True)

directory = "D:\GitHub\point\data"
np.save(directory + "\data_synth_points.npy", data.locs)
np.save(directory + "\data_synth_sizes.npy", data.sizes)
np.save(directory + "\data_synth_variables.npy", data.variables)
np.save(directory + "\data_synth_space.npy", data.space)
