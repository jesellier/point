import numpy as np
import arrow
import time

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from point.misc import Space, TensorMisc
from point.point_process import PointsData

import gpflow
from gpflow.config import default_float

from vbpp.model import VBPP


################LOAD SYNTH DATA
directory = "D:\GitHub\point\data\data_rff"
expert_seq = np.load(directory + "\data_synth_points.npy")
expert_variables = np.load(directory + "\data_synth_variables.npy", allow_pickle=True)
expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)
expert_sizes = np.load(directory + "\data_synth_sizes.npy", allow_pickle=True)
expert_data = PointsData(expert_sizes, expert_seq, expert_space, expert_variables)
rng = np.random.RandomState()
space = Space(expert_space)

####### INIT PARAMETERS
tf.random.set_seed(10)
emp = 2 * np.sqrt(np.mean(expert_sizes) / space.measure) / 3
variance = tf.Variable([emp**2], name='sig')
length_scale = tf.Variable([0.2], dtype= default_float(), name='lengthscale')


initial_learning_rate = 1.0
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps= 5,
    decay_rate=0.8,
    staircase=True
    )
    
#optimizer = tf.keras.optimizers.SGD(learning_rate= lr_schedule )
optimizer = tf.keras.optimizers.SGD(learning_rate= 1.0 )


def domain_grid(space, step = 0.05):
    x = y = np.arange(space._lower_bound, space._higher_bound, step)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((X.shape[0]**2,2))
    Z[:,0] = np.ravel(X)
    Z[:,1] = np.ravel(Y)
    return Z


def build_model(events, space):
    kernel = gpflow.kernels.SquaredExponential()
    Z = domain_grid(space, step = 0.5)
    M = Z.shape[0]

    feature = gpflow.inducing_variables.InducingPoints(Z)
    q_mu = np.zeros(M)
    q_S = np.eye(M)
    num_events = len(events)
    #beta0 = np.sqrt(num_events / space.measure)
    model = VBPP(feature, kernel, space.bounds, q_mu, q_S, beta0=0.5, num_events=num_events)
    #model = VBPP(feature, kernel, space.bounds, q_mu, q_S, num_events=num_events)
    gpflow.set_trainable(model.beta0, False)
    return model



####### HYPER PARAMETERS
num_epochs = 100
n_batch = 500

t0 = time.time()
space = Space(expert_space)
rng = np.random.RandomState(10)
variables = build_model(expert_data[0], space).trainable_variables
options = dict(maxiter = 1)


def compute_loss_and_gradients(loss_closure, variables):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        loss = loss_closure()
    grads = tape.gradient(loss, variables)
    return loss, grads

for epoch in range(num_epochs):

    batch_ids = expert_data.shuffled_index(n_batch, random_state = rng)
    grads = TensorMisc().pack_tensors_to_zeros(variables)
    num_grads = 0
    t0 = time.time()
    
    print("")
    print("start [%s] %d-th epoch : " % \
            (arrow.now(), epoch+1))

    for event_id in batch_ids :
        
        events = expert_data[event_id]
        model = build_model(events, space)
        TensorMisc().assign_tensors(model.trainable_variables, variables)

        def objective_closure():
            return -model.elbo(events)
        
        try :
            _, grad_event = compute_loss_and_gradients(objective_closure, model.trainable_variables)
            grad_event = TensorMisc().pack_tensors(grad_event)   
            grads += grad_event
        except:
            print("[{}] ERROR: iteration skipped to id : {}".format(arrow.now(), event_id))

    grads = grads / n_batch
    grads = TensorMisc().unpack_tensors(model.trainable_variables, grads)
    grads = tf.clip_by_global_norm(grads, clip_norm = 1, use_norm=None, name=None)[0]
    optimizer.apply_gradients(zip(grads, variables))
    
    TensorMisc().assign_tensors(model.trainable_variables, variables)
    parameters = model.parameters
    
    #printout
    #print("[{}] grads : {}".format(arrow.now(), grads))
    print("[{}] learning_rate : {}".format(arrow.now(), optimizer._decayed_lr(tf.float32)))
    print("[{}] new variables : {}".format(arrow.now(), parameters[-2:]))
    print("[{}] time : {}".format(arrow.now(), time.time() - t0))

        
#directory = "D:\GitHub\point\results"
#np.save(directory + "\data_vbb.npy", out)


        





