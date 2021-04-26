import numpy as np
import arrow
import sys

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from point.point_process import CoxLowRankSpatialModel, Space

rng = np.random.RandomState(20)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD SYNTH DATA

directory = "D:\GitHub\point\data"
expert_seq = np.load(directory + "\data_synth_points.npy")
expert_variables = np.load(directory + "\data_synth_variables.npy", allow_pickle=True)
expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)
print("[{}] SYNTH-DATA variables : {}".format(arrow.now(), expert_variables), file=sys.stderr)

####### HYPER PARAMETERS
num_epochs = 1
num_batches = 1

batch_learner_size = 10
batch_expert_size = None
n_components = 800 #RFF sampling order

####### INIT
num_experts = expert_seq.shape[0]
    
#variance = tf.Variable(tf.random.uniform(shape = [1], minval=0, maxval = 10, dtype=float_type), name='sig')
#length_scale = tf.Variable(tf.random.uniform(shape = [2], minval=0, maxval = 1, dtype=float_type), dtype=float_type, name='l')

variance = tf.Variable([20.0], dtype=float_type, name='sig')
length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='l')

space = Space(lower_bounds = expert_space[0], higher_bounds = expert_space[1])
model = CoxLowRankSpatialModel(length_scale=length_scale, variance = variance, n_components = n_components, random_state = rng)
reward_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=None, length_scale= tf.constant(0.5,  dtype=float_type),name='ExponentiatedQuadratic')

learning_rate = 0.001
optimizer = tf.keras.optimizers.SGD(learning_rate= learning_rate)


print("[{}] INIT with variance:={}, length_scale:={}".format(arrow.now(), variance, length_scale))

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

batch_expert_locs = expert_seq

for epoch in range(num_epochs):

    for b in range(num_batches) :
        
        print("")
        print("start [%s] %d-th epoch - %d batch : " % \
                  (arrow.now(), epoch+1, b+ 1))
        
        ## shuffle batch expert
        shuffled_ids = np.arange(num_experts)
        
        if batch_expert_size is not None and batch_expert_size < num_experts :
            rng.shuffle(shuffled_ids)
            shuffled_ids = shuffled_ids[- batch_expert_size :]
            #batch_expert_sizes = expert_sizes[shuffled_ids]
            batch_expert_locs = expert_seq[shuffled_ids]

        # generate learner samples
        learner_data = model.generate(sp = space, batch_size = batch_learner_size, calc_grad = True, verbose = verbose)
        batch_learner_locs = learner_data._locs
        
        # compute rewards
        rewards_per_points = reward_vector(learner_data._locs, batch_expert_locs, reward_kernel)
        rewards_per_batch =[ tf.math.reduce_sum(s) for s in tf.split(rewards_per_points, learner_data._sizes )]
        rewards_per_batch = tf.stack(rewards_per_batch)
        
        #compute gradient
        grad_per_batch = learner_data._grad
        grads =[ tf.math.reduce_sum(tf.linalg.diag(rewards_per_batch) @  tf.stack(c) / batch_learner_size , axis = 0)  for c in grad_per_batch.T]
        
        #compute log likelihood
        loglik =  tf.math.reduce_sum(tf.stack(learner_data._loglik) / batch_learner_size, axis = 0)
        loglik = loglik.numpy()
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        #printout
        print("[{}] grad : {}".format(arrow.now(), grads[0]))
        print("[{}] grad : {}".format(arrow.now(), grads[1]))
        print("[{}] variables : {}".format(arrow.now(), model.trainable_variables))
        print("[{}] loss : {}".format(arrow.now(), loglik))



