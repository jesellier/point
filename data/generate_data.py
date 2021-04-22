import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.point_process import CoxLowRankSpatialPP



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng = np.random.RandomState(40)

variance = tf.Variable(100.0, dtype=float_type, name='sig')
length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        
p = CoxLowRankSpatialPP(length_scale=length_scale, variance = variance, random_state = rng)
data, sizes = p.generate(batch_size  = 1000)

directory = "D:\GitHub\point\data"
np.save(directory + "\data-synth_len.npy", sizes)
np.save(directory + "\data-synth_loc.npy", data)