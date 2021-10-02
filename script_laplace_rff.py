
import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import gpflow
from gpflow.config import default_float

from point.laplace import LaplaceApproximation
from point.helper import get_process, method
from point.misc import Space
from point.metrics import Evaluation

from data_2D import build_data_1000, build_data_400,  domain_grid, print_grid, get_synthetic_generative_model, lambda_synth_1000, lambda_synth_400



def get_rff_model(space):

    variance = tf.Variable(2.5, dtype=default_float(), name='sig')
    length_scale = tf.Variable([0.5,0.5], dtype=default_float(), name='lenght_scale')
    n_components = 250

    model = get_process(length_scale = length_scale, 
                variance = variance, 
                method = method.RFF_NO_OFFSET,
                space = space,
                n_components = n_components, 
                random_state = rng)

    return LaplaceApproximation(model) 


def optim_func(model, verbose = False):

    t0 = time.time()
    
    model._optimize_mode(optimizer = "scipy", maxiter = 50)
    #model._optimize_mode(optimizer = "line_search", n_seeds = 100, restarts= 1, maxiter = 50, verbose = False) 
    
    for _ in range(1):
        model.optimize(optimizer = "scipy_autodiff",  maxiter = 50)
        model._optimize_mode(optimizer = "scipy", maxiter = 50) 

    if verbose :
        print("TOTAL finished_in := [%f] " % (time.time() - t0))
        print(model.log_posterior())
        print(model.log_marginal_likelihood())
        print(model.lrgp.trainable_parameters)





#%% eval model
rng  = np.random.RandomState(10)
#lp.lrgp.variance.assign(22)

lambda_truth, grid, space = lambda_synth_1000()
gpp = get_synthetic_generative_model(lambda_truth, grid, random_state = rng)
lp = get_rff_model(space)

#adjsutement
lp.model.lrgp._G = tf.Variable(lp.model.lrgp._G)
#gpflow.set_trainable(lp.lrgp.variance, False)

#evaluation
evl = Evaluation(lp, gpp)
evl.run(optim_func, n_samples = 10)
res_r = evl.results
print(evl.results)


#%% fit and print model
rng  = np.random.RandomState(10)

lambda_truth, grid, space = lambda_synth_1000()
gpp = get_synthetic_generative_model(lambda_truth, grid, random_state = rng)
lp = get_rff_model(space)

#adjustement
lp.model.lrgp._G = tf.Variable(lp.model.lrgp._G)
X = gpp.generate()
lp.set_X(X)
optim_func(lp, verbose = True)

lambda_mean = lp.predict_lambda(grid) 
print_grid(grid, lambda_mean)
print_grid(grid, lambda_truth)
#print_grid(grid, lambda_truth, X)

 


#%% gen intensity

rng  = np.random.RandomState(10)
X, bound = build_data_1000()
lp = get_rff_model(space)

#adjsutement
lp.model.lrgp._G = tf.Variable(lp.model.lrgp._G)
lp.set_X(X)
optim_func(lp, verbose = True)

grid, x_mesh, y_mesh = domain_grid(bound = bound, step = 0.1)
lambda_mean = lp.predict_lambda(grid) 
print_grid(grid, lambda_mean)

import matplotlib.pyplot as plt
plt.xlim(grid.min(), grid.max())
plt.plot(X[:,0], X[:,1],'ro', markersize = 0.5)
plt.show()

 
#%% write to CSV
import csv
#lambda_v = ***
a = np.vstack((grid[:,0], grid[:,1], lambda_v[:,0])).T
directory = "D:\GitHub\point\data\data_synth_1000_2.csv"
with open(directory, 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(a)
    
