import numpy as np
import arrow

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import gpflow
from gpflow.config import default_float

from point.misc import Space, TensorMisc
from point.reward import Reward
from point.point_process import PointsData
from point.helper import get_process, method
from point.optim.optim_scipy import OptimScipy, initial_parameters

rng = np.random.RandomState()


################LOAD SYNTH DATA
directory = "D:\GitHub\point\data\data_rff"
expert_seq = np.load(directory + "\data_synth_points.npy")
expert_variables = np.load(directory + "\data_synth_variables.npy", allow_pickle=True)
expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)
expert_sizes = np.load(directory + "\data_synth_sizes.npy", allow_pickle=True)

expert_data = PointsData( expert_sizes, expert_seq, expert_space)
space = Space(expert_space)

####### INIT PARAMETERS
tf.random.set_seed(10)
lambda_estimate = 2 * np.sqrt(np.mean(expert_sizes) / space.measure) / 3
variance = tf.Variable([lambda_estimate**2], name='sig')
length_scale = tf.Variable(tf.random.uniform(shape = [2], minval=0, maxval = 1, dtype= default_float()), name='lengthscale')
beta0 = tf.Variable([0.5], dtype=default_float(), name='beta0')

#variance = tf.Variable([8.0], dtype=default_float(), name='sig')
#length_scale = tf.Variable([0.2], dtype=default_float(), name='lengthscale')
#beta0 = tf.Variable([0.5], dtype=default_float(), name='beta0')

######## INSTANTIATE MODEL
method = method.RFF
model = get_process(method, space = space, beta0 = beta0, n_components = 250, random_state = rng, 
                    variance = variance, length_scale = length_scale)
gpflow.set_trainable(model.lrgp.beta0, False)

######## LEARNING HYPER
reward_kernel = tfk.ExponentiatedQuadratic(amplitude=None, length_scale= tf.constant(0.5,  dtype=default_float()),name='ExponentiatedQuadratic')




####### HYPER PARAMETERS
batch_learner_size = 100

print("[{}] SYNTH-DATA variables : {}".format(arrow.now(), expert_variables))
print("")
print("[{}] INIT with beta0:={}, length_scale:={}, variance:={}".format(arrow.now(), beta0, length_scale, variance))
print("")


variables = model.trainable_variables
x = initial_parameters(variables)

values = TensorMisc().unpack_tensors(variables, x)
TensorMisc().assign_tensors(variables, values)

batch_expert_locs = expert_seq


def _compute_loss_and_gradients():
    
    learner_data = model.generate(batch_size = batch_learner_size, calc_grad = True, verbose = False)

    # compute rewards
    rewards_per_points = Reward().rewards(learner_data.locs, batch_expert_locs, reward_kernel)
    rewards_per_batch =[ tf.math.reduce_sum(s) for s in tf.split(rewards_per_points, learner_data.sizes )]
    rewards_per_batch = tf.stack(rewards_per_batch)
    rewards_per_batch = tf.math.l2_normalize(rewards_per_batch)
    loss = sum(rewards_per_batch) / batch_learner_size

    #compute gradient
    grad_per_batch = learner_data.grad
    #rewards_weights = tf.math.l2_normalize(rewards_per_batch)
    rewards_weights = rewards_per_batch
    grads = tf.math.reduce_sum(tf.linalg.diag(rewards_weights) @  grad_per_batch / batch_learner_size , axis = 0)
 
    # mulitply by -1 [to maximize] and normalize [to avoid explosion]
    grads = -1 * grads
    grads = TensorMisc().unpack_tensors(model.trainable_variables, grads)
    #grads = tf.clip_by_global_norm(grads, clip_norm = 1, use_norm=None, name=None)[0]

    return loss, grads

    
results = OptimScipy().minimize(closure = _compute_loss_and_gradients, variables = model.trainable_variables,compile = False)






























