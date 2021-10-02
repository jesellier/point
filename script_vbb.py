# Copyright (C) Secondmind Ltd 2017-2020
#
# Licensed under the Apache License, Version 2.0

# run from this file's directory


import numpy as np
import time

import tensorflow as tf
import gpflow
from gpflow.config import default_float

from vbpp.model import VBPP
from vbpp.scipy import Scipy

from point.misc import Space
from point.metrics import Evaluation

from data_2D import build_data_1000, build_data_400,  domain_grid, print_grid, get_synthetic_generative_model, lambda_synth_1000, lambda_synth_400



def build_model(space, events):
    
    kernel = gpflow.kernels.SquaredExponential()

    num_inducing = 250
    domain_area = space.measure
    step = np.around((np.sqrt(domain_area) / (np.sqrt(num_inducing) - 1)), 2)

    Z, _, _ = domain_grid(step = step)
    M = Z.shape[0]

    feature = gpflow.inducing_variables.InducingPoints(Z)
    q_mu = np.zeros(M)
    
    q_S = np.eye(M)
    num_events = events.shape[0]
    beta0 = np.sqrt(num_events / domain_area)
    model = VBPP(feature, kernel,space.bound2D, q_mu, q_S, beta0=beta0, num_events=num_events)
    #gpflow.set_trainable(model.beta0, False)
    
    return model


class VBPPAdapter():
    
    def __init__(self, model, X = None):
        
        if not isinstance(model, VBPP):
            raise ValueError("'model' must be a 'VBPP' instance")
        
        self.model = model
        self._X = X
        
    def predict_lambda(self, Xnew):
        out, _, _ = self.model.predict_lambda_and_percentiles(Xnew)
        return out
    
    def set_X(self, X):
        self._X = X
        pass
    
    def log_likelihood(self, X_new):
        K = gpflow.covariances.Kuu(self.model.inducing_variable, self.model .kernel)
        integral_term = self.model._elbo_integral_term(K)
        data_term = self.model._elbo_data_term(X_new, K)
        out = self.model.num_observations * integral_term + data_term
        return out

        


def optim_func(adapter, verbose = False):
    
    maxiter = 100
    events = adapter._X

    def objective_closure():
        return -adapter.model.elbo(events)

    Scipy().minimize(objective_closure, adapter.model.trainable_variables, options = {'maxiter': maxiter}, compile = False)



#%% gen model
rng  = np.random.RandomState(10)

lambda_truth, grid, space = lambda_synth_1000()
gpp = get_synthetic_generative_model(lambda_truth, grid, random_state = rng)
X_init = gpp.generate()
vbpp = build_model(space, X_init)
vbpp_adapter = VBPPAdapter(vbpp)

#evaluation
evl = Evaluation(vbpp_adapter, gpp)
evl.run(optim_func, n_samples = 10)
res_v = evl.results
print(evl.results)


#%%
rng  = np.random.RandomState(10)

lambda_truth, grid, space = lambda_synth_1000()
gpp = get_synthetic_generative_model(lambda_truth, grid, random_state = rng)
vbpp = build_model(space, gpp.generate())
vbpp_adapter = VBPPAdapter(vbpp)

#adjsutement
X = gpp.generate()
vbpp_adapter.set_X(X)

optim_func(vbpp_adapter, verbose = False)
lambda_mean = vbpp_adapter.predict_lambda(grid) 
print_grid(grid, lambda_mean)
print_grid(grid, lambda_truth)
print_grid(grid, lambda_truth, X)
