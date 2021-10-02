
import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from gpflow.config import default_float

from point.laplace import LaplaceApproximation
from point.helper import get_process, method
from point.metrics import Evaluation

from data_2D import build_data_1000, build_data_400,  domain_grid, print_grid, get_synthetic_generative_model, lambda_synth_400,lambda_synth_1000


rng  = np.random.RandomState(15)



def get_nyst_model(space):

    variance = tf.Variable(2.5, dtype=default_float(), name='sig')
    length_scale = tf.Variable([0.5,0.5], dtype=default_float(), name='lenght_scale')
    n_components = 50

    model = get_process(length_scale = length_scale, 
                    variance = variance, 
                    method =  method.NYST_DATA,
                    space = space,
                    n_components = n_components, 
                    random_state = rng)
    
    lp = LaplaceApproximation(model) 
    
    return lp


def optim_func(model, verbose = False):

    model.lrgp.set_data(model._X)
    model.lrgp.fit()
    t0 = time.time()
    model._optimize_mode(optimizer = "scipy", maxiter = 50) 

    if verbose :
        print("TOTAL finished_in := [%f] " % (time.time() - t0))
        print(model.log_posterior())
        print(model.log_marginal_likelihood())
        print(model.lrgp.trainable_parameters)




#%% gen model
rng  = np.random.RandomState(10)

lambda_truth, grid, space = lambda_synth_1000()
gpp = get_synthetic_generative_model(lambda_truth, grid, random_state = rng)
lp = get_nyst_model(space)

#evaluation
evl = Evaluation(lp, gpp)
evl.run(optim_func, n_samples = 10)
res_n = evl.results
print(evl.results)


#print_grid(grid, lambda_mean, X_train)


#%% fit and print model
rng  = np.random.RandomState(10)

lambda_truth, grid, space = lambda_synth_1000()
gpp = get_synthetic_generative_model(lambda_truth, grid, random_state = rng)
lp = get_nyst_model(space)

#adjsutement
#X = gpp.generate()
lp.set_X(X)
optim_func(lp, verbose = False)

lambda_mean = lp.predict_lambda(grid) 
print_grid(grid, lambda_mean)

