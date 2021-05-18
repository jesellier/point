import numpy as np
import arrow
import time

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import gpflow
from gpflow.config import default_float

from point.misc import Space, TensorMisc
from point.point_process import PointsData
from point.helper import get_process, method
from point.reward import Reward

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
length_scale = tf.Variable(tf.random.uniform(shape = [1], minval=0, maxval = 1, dtype= default_float()), name='lengthscale')
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

initial_learning_rate = 1.0
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps= 5,
    decay_rate=0.8,
    staircase=True
    )
    
optimizer = tf.keras.optimizers.SGD(learning_rate= lr_schedule )
#optimizer = tf.keras.optimizers.Adam(learning_rate= lr_schedule, beta_1=0.6, beta_2=0.4, epsilon=1e-07 )



####### HYPER PARAMETERS
num_epochs = 1
num_iter = 100
batch_learner_size = 50
batch_expert_size = None
########################

print("[{}] SYNTH-DATA variables : {}".format(arrow.now(), expert_variables))
print("")
print("[{}] INIT with beta0:={}, length_scale:={}, variance:={}".format(arrow.now(), beta0, length_scale, variance))



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
        if batch_expert_size is not None and batch_expert_size < expert_data.num_sequences :
            shuffled_ids = np.arange(expert_data.num_sequences)
            rng.shuffle(shuffled_ids)
            shuffled_ids = shuffled_ids[- batch_expert_size :]
            batch_expert_locs = expert_seq[shuffled_ids]

        # generate learner samples
        learner_data = model.generate(batch_size = batch_learner_size, calc_grad = True, verbose = False)
        
        # compute rewards
        rewards_per_batch =  Reward().rewards_per_batch(learner_data, batch_expert_locs, reward_kernel)
        loss = sum(rewards_per_batch) / batch_learner_size # := minus reward term
        
        #compute gradient
        grad_per_batch = learner_data.grad
        rewards_weights = tf.math.l2_normalize(rewards_per_batch)
        grads = tf.math.reduce_sum(tf.linalg.diag(rewards_weights) @  grad_per_batch / batch_learner_size , axis = 0)
 
        # mulitply by -1 [to maximize] and normalize [to avoid explosion]
        grads = -1 * grads
        grads = TensorMisc().unpack_tensors(model.trainable_variables, grads)
        grads = tf.clip_by_global_norm(grads, clip_norm = 1, use_norm=None, name=None)[0]

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #printout
        print("[{}] Total.loss : {}".format(arrow.now(), loss))
        print("[{}] grads : {}".format(arrow.now(), grads))
        print("[{}] learning_rate : {}".format(arrow.now(), optimizer._decayed_lr(tf.float32)))
        print("[{}] new variables : {}".format(arrow.now(), model.parameters))

print("time:" + str(time.time() - t0))



