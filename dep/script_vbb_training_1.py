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
from vbpp.scipy import Scipy



################LOAD SYNTH DATA
directory = "D:\GitHub\point\data\data_rff"
expert_seq = np.load(directory + "\data_synth_points.npy")
expert_variables = np.load(directory + "\data_synth_variables.npy", allow_pickle=True)
expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)
expert_sizes = np.load(directory + "\data_synth_sizes.npy", allow_pickle=True)
expert_data = PointsData( expert_sizes, expert_seq, expert_space, expert_variables)
rng = np.random.RandomState()
space = Space(expert_space)

####### INIT PARAMETERS
tf.random.set_seed(10)
emp = 2 * np.sqrt(np.mean(expert_sizes) / space.measure) / 3
variance = tf.Variable([emp**2], name='sig')
length_scale = tf.Variable([0.2], dtype= default_float(), name='lengthscale')

####### INIT PARAMETERS
variance = tf.Variable([8.0], dtype= default_float(), name='sig')
length_scale = tf.Variable([0.2], dtype= default_float(), name='lengthscale')


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
    gpflow.set_trainable(model.beta0, False)
    return model



####### HYPER PARAMETERS
num_epochs = 1
n_batch = 1

t0 = time.time()
space = Space(expert_space)
rng = np.random.RandomState()
variables = build_model(expert_data[0], space).trainable_variables
options = dict(maxiter = 1)
results_lst = []


for epoch in range(num_epochs):

    batch_ids = expert_data.shuffled_index(n_batch, random_state = rng)
    batch_iteration = 0

    for event_id in batch_ids :
        
        print("")
        t0 = time.time()
        print("start [%s] %d-th epoch - %d iteration : " % \
                  (arrow.now(), epoch+1, batch_iteration+1))
            
        events = expert_data[event_id]
        model = build_model(events, space)

        def objective_closure():
            return -model.elbo(events)

        try :
            Scipy().minimize(closure = objective_closure, variables = model.trainable_variables, compile = False, options= options)
            #Scipy().minimize(closure = objective_closure, variables = model.trainable_variables, compile = False)
            params = TensorMisc().pack_tensors(model.kernel.parameters)
            results_lst.append(params)
            print("[{}] SUCCESS with new variables : {}".format(arrow.now(), model.kernel.parameters))
            print("[{}] time : {}".format(arrow.now(), time.time() - t0))
        except:
            print("[{}] ERROR: iteration skipped to : {}".format(arrow.now(), model.kernel.parameters))

        batch_iteration +=1
        
out = tf.stack(results_lst, axis = 0)
average_params = tf.math.reduce_mean(out, axis = 0)

#directory = "D:\GitHub\point\results"
#np.save(directory + "\data_vbb.npy", out)


        





