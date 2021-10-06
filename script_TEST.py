
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

from data_2D import build_data_1000, build_data_400,  domain_grid, print_grid, get_synthetic_generative_model, lambda_synth_1000, lambda_synth_400

import theano
import theano.tensor as tt

import arviz as az
theano.config.compute_test_value = "ignore"

import pymc3 as pm

import seaborn as sb
import matplotlib.pyplot as plt



def get_rff_model(space):

    variance = tf.Variable(2.5, dtype=default_float(), name='sig')
    length_scale = tf.Variable([0.5,0.5], dtype=default_float(), name='lenght_scale')
    n_components = 25

    model = get_process(length_scale = length_scale, 
                variance = variance, 
                method = method.RFF_NO_OFFSET,
                space = space,
                n_components = n_components, 
                random_state = rng)

    return LaplaceApproximation(model) 


def optim_func(model, verbose = False):

    t0 = time.time()
    
    model._optimize_mode(optimizer = "scipy", maxiter = 2)
    #model._optimize_mode(optimizer = "line_search", n_seeds = 100, restarts= 1, maxiter = 50, verbose = False) 
    
    for _ in range(1):
        model.optimize(optimizer = "scipy_autodiff",  maxiter = 2)
        model._optimize_mode(optimizer = "scipy", maxiter = 2) 

    if verbose :
        print("TOTAL finished_in := [%f] " % (time.time() - t0))
        print(model.log_posterior())
        print(model.log_marginal_likelihood())
        print(model.lrgp.trainable_parameters)



#fit model
rng  = np.random.RandomState(10)

lambda_truth, grid, space = lambda_synth_400()
gpp = get_synthetic_generative_model(lambda_truth, grid, random_state = rng)
lp = get_rff_model(space)

#adjustement
lp.lrgp._G = tf.Variable(lp.lrgp._G)
X = gpp.generate()
lp.set_X(X)
optim_func(lp, verbose = True)


###############################################################################################
beta = 1e-06

def likelihood_func(x, args):
    (M,F) = args
    f = F.dot(x) + beta
    out = - (x * M.dot(x)).sum() + np.log(f**2).sum()
    return out

def gradient_func(x, args):
    (M,F) = args 
    f = F.dot(x) + beta
    out = - (M.T + M).dot(x) + 2 * sum( F / np.expand_dims(f, 1))
    return out


def trace_plot(trace):
    plt.figure(figsize=(15, 4))
    plt.axhline(0.7657852, lw=2.5, color="gray")
    plt.plot(trace, lw=2.5)
    plt.ylabel('latent')
    plt.xlabel("Iteration")
    plt.title("trace_plot")
    

class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector]  
    otypes = [tt.dscalar]  

    def __init__(self, lfunc, gfunc, args):
        
        self.lfunc = lfunc
        self.args = args
        self.logpgrad = LogLikeGrad(gfunc, args)
        
    def perform(self, node, inputs, outputs):
        x = inputs[0]
        outputs[0][0] = self.lfunc(x, self.args)
 

    def grad(self, inputs, output_grads):
        x = inputs[0]
        g = self.logpgrad(x)
        return [output_grads[0] *g]


class LogLikeGrad(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, gfunc, args):
        self.args = args
        self.gfunc = gfunc

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        outputs[0][0] = self.gfunc(x, self.args)



discard_tuned_samples= True
n = lp.n_components
F = lp.lrgp.feature(X).numpy()
M = lp.lrgp.M().numpy()

args = (M,F)
logl = LogLikeWithGrad(likelihood_func, gradient_func, args)


with pm.Model() as Centered_eight:
    x  = pm.Normal('w', mu= 0, sigma=1, shape= n )  
    pm.Potential("likelihood", logl(x))
    
    trace = pm.sample(10, tune = 10, cores = 1, chains=2, 
                      compute_convergence_checks = True, 
                      return_inferencedata=True,
                      discard_tuned_samples = discard_tuned_samples
                      )

az.plot_trace(trace)
s = az.summary(trace)



