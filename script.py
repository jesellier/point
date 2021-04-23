import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng = np.random.RandomState(20)

#variance = tf.Variable(100.0, dtype=float_type, name='sig')
#length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        

directory = "D:\GitHub\point\data"
sizes = np.load(directory + "\data-synth_len.npy")
seq_l = np.load(directory + "\data-synth_loc.npy")

n_batch = seq_l.shape[0]

shuffled_ids = np.arange(n_batch)
rng.shuffle(shuffled_ids)

n_expert = 1000
n_learn = 10
shuffled_expert_ids = shuffled_ids[:n_expert]
shuffled_learner_ids  = shuffled_ids[-n_learn:]

batch_learner_s = sizes[shuffled_learner_ids]
batch_learner_l = seq_l[shuffled_learner_ids]

batch_expert_s = sizes[shuffled_expert_ids]
batch_expert_l = seq_l[shuffled_expert_ids]

learner_l = tf.constant(batch_learner_l,  dtype=float_type)  
expert_l =  tf.constant(batch_expert_l,  dtype=float_type) 


length_scale = tf.constant(0.5,  dtype=float_type)
kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
    amplitude=None, length_scale= length_scale, validate_args=False,
    name='ExponentiatedQuadratic'
)


def concenat_vector(vec):
    cont_vec =  tf.unstack(vec, axis = 0)  
    cont_vec =  tf.concat(cont_vec, axis=0)  
    cont_vec = tf.boolean_mask(cont_vec, (cont_vec[:,0] != 0))
    return cont_vec
    

########## Compute REWARD VECTOR
def reward_vector(learner_l, expert_l, kernel):
    learner_l = concenat_vector(learner_l)
    expert_l = concenat_vector(expert_l) 

    mat_el = kernel.matrix(expert_l, learner_l)
    mat_ll = kernel.matrix(learner_l, learner_l)
    
    sum_el = tf.math.reduce_sum(mat_el , axis = 0)
    sum_ll = tf.math.reduce_sum(mat_ll , axis = 0)
    reward = sum_el / n_expert  - sum_ll / n_learn
    return reward

reward = reward_vector(learner_l, expert_l, kernel)
size_l = sizes[shuffled_learner_ids]
splits = tf.split(reward, size_l)
rewards =[ tf.math.reduce_sum(s).numpy() for s in splits]

print(rewards)





