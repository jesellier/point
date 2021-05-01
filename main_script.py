import numpy as np
import arrow
import sys
import time

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.point_process import CoxLowRankSpatialModel, Space
from point.low_rank_rff import LowRankRFF
from point.low_rank_nystrom import LowRankNystrom
from point.helper import get_process, method

import gpflow.kernels as gfk

#LOAD SYNTH DATA
directory = "D:\GitHub\point\data"
expert_seq = np.load(directory + "\data_synth_points.npy")
expert_variables = np.load(directory + "\data_synth_variables.npy", allow_pickle=True)
expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)
print("[{}] SYNTH-DATA variables : {}".format(arrow.now(), expert_variables), file=sys.stderr)

rng = np.random.RandomState()

####### HYPER PARAMETERS
num_epochs = 1
num_iter = 1000

batch_learner_size = 50
batch_expert_size = 50
n_components = 250 

####### INIT
num_experts = expert_seq.shape[0]
    
#tf.random.set_seed(10)
#variance = tf.Variable(tf.random.uniform(shape = [1], minval=0, maxval = 10, dtype=float_type), name='sig')
#length_scale = tf.Variable(tf.random.uniform(shape = [2], minval=0, maxval = 1, dtype=float_type), dtype=float_type, name='lengthscale')

variance = tf.Variable([2.0], dtype=float_type, name='sig')
length_scale = tf.Variable([0.2,0.8], dtype=float_type, name='l')

######## INSTANTIATE MODEL

space = Space(lower_bounds = expert_space[0], higher_bounds = expert_space[1])
method = method.RFF
model = get_process(length_scale, variance, method, n_components = 250, random_state = rng )

######## LEARNING HYPER
reward_kernel = tfk.ExponentiatedQuadratic(amplitude=None, length_scale= tf.constant(0.5,  dtype=float_type),name='ExponentiatedQuadratic')

initial_learning_rate = 0.8
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps= 20,
    decay_rate=0.9,
    staircase=True
    )
    
optimizer = tf.keras.optimizers.SGD(learning_rate= lr_schedule )

print("")
print("[{}] INIT with length_scale:={}, variance:={}".format(arrow.now(), length_scale, variance))

def concenat_vector(vec):
    cont_vec =  tf.unstack(vec, axis = 0)  
    cont_vec =  tf.concat(cont_vec, axis=0)  
    cont_vec = tf.boolean_mask(cont_vec, (cont_vec[:,0] != 0))
    return cont_vec


def reward_vector(learner_l, expert_l, kernel):
    
    n_learner_batches = len(learner_l)
    n_expert_batches = len(expert_l)

    learner_l = concenat_vector(learner_l)
    expert_l = concenat_vector(expert_l) 
    
    mat_el = kernel.matrix(expert_l, learner_l)
    mat_ll = kernel.matrix(learner_l, learner_l)

    sum_el = tf.math.reduce_sum(mat_el , axis = 0)
    sum_ll = tf.math.reduce_sum(mat_ll , axis = 0)
    reward = sum_el / n_expert_batches  - sum_ll / n_learner_batches
    
    return reward


############# MAIN TRAINING LOOP
verbose = True
store_values = []

batch_expert_locs = expert_seq

t0 = time.time()
for epoch in range(num_epochs):

    for i in range(num_iter) :
        
        print("")
        print("start [%s] %d-th epoch - %d iteration : " % \
                  (arrow.now(), epoch+1, i + 1))
        
        ## shuffle batch expert
        shuffled_ids = np.arange(num_experts)
        
        if batch_expert_size is not None and batch_expert_size < num_experts :
            rng.shuffle(shuffled_ids)
            shuffled_ids = shuffled_ids[- batch_expert_size :]
            batch_expert_locs = expert_seq[shuffled_ids]

        # generate learner samples
        learner_data = model.generate(space = space, batch_size = batch_learner_size, calc_grad = True, verbose = False)
        batch_learner_locs = learner_data.locs
        
        # compute rewards
        rewards_per_points = reward_vector(learner_data.locs, batch_expert_locs, reward_kernel)
        rewards_per_batch =[ tf.math.reduce_sum(s) for s in tf.split(rewards_per_points, learner_data.sizes )]
        rewards_per_batch = tf.stack(rewards_per_batch)
        rewards_per_batch = tf.math.l2_normalize(rewards_per_batch)
        
        #compute gradient
        grad_per_batch = learner_data.grad
        grads =[ tf.math.reduce_sum(tf.linalg.diag(rewards_per_batch) @  tf.stack(c) / batch_learner_size , axis = 0)  for c in grad_per_batch.T]
        
        # mulitply by -1 [to maximize] and normalize [to avoid explosion]
        grads = [-1 * g for g in grads]
        grads = tf.clip_by_global_norm(grads, 1, use_norm=None, name=None)[0]
        
        #grads[1] = tf.constant([0.0], dtype=float_type)
        #grads[0] = tf.constant([0.0,0.0], dtype=float_type)
  
        #compute log likelihood
        #loglik =  tf.math.reduce_sum(tf.stack(learner_data._loglik) / batch_learner_size, axis = 0)
        #loglik = loglik.numpy()

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #store_values.append(model.trainable_variables[0].numpy()[0])

        #printout
        print("[{}] grad0 : {}".format(arrow.now(), grads[0]))
        print("[{}] grad1 : {}".format(arrow.now(), grads[1]))
        print("[{}] learning_rate : {}".format(arrow.now(), optimizer._decayed_lr(tf.float32)))
        print("[{}] new variables : {}".format(arrow.now(), model.parameters))
        #print("[{}] loss : {}".format(arrow.now(), loglik))

print(time.time() - t0)
#estimated_variance = np.mean(store_values[-10:])




