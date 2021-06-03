import numpy as np
import arrow
import time

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


import matplotlib.pyplot as plt
import gpflow

from point.misc import Space, TensorMisc
from point.point_process import PointsData

import gpflow
from gpflow.config import default_float
import gpflow.kernels as gfk

from vbpp.model import VBPP
from vbpp.scipy import Scipy


################LOAD SYNTH DATA

def build_data():
    directory = "D:\GitHub\point\data"
    expert_seq = np.load(directory + "\data_synth_points.npy")
    expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)

    return expert_seq[0, :, :], expert_space


def domain_grid(space = (-1,1) , step = 0.05):
    x = y = np.arange(space[0], space[1] + step, step)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((X.shape[0]**2,2))
    Z[:,0] = np.ravel(X)
    Z[:,1] = np.ravel(Y)

    return Z, x, y


def domain_area(domain):
    return np.prod(domain.max(0) - domain.min(0))


def build_model(events, domain):
    kernel = gpflow.kernels.SquaredExponential()
    
    num_inducing = 20
    step = int((domain.max(0) - domain.min(0) ) / (np.sqrt(num_inducing) - 1))
    
    Z, _, _ = domain_grid(space = domain, step = step)
    n = Z.shape[0]

    feature = gpflow.inducing_variables.InducingPoints(Z)
    q_mu = np.zeros(n)
    q_S = np.eye(n)
    num_events = len(events)
    beta0 = np.sqrt(num_events / domain_area(domain))
    
    domain2D = np.array([domain, domain])
    model = VBPP(feature, kernel, domain2D, q_mu, q_S, beta0=beta0, num_events=num_events)
    return model


def demo():
    events, domain = build_data()
    model = build_model(events, domain)

    def objective_closure():
        return -model.elbo(events)

    Scipy().minimize(objective_closure, model.trainable_variables, compile = False)

    X, x_mesh, y_mesh = domain_grid(space = domain, step = 0.5)
    lambda_mean = model.predict_lambda(X)
    
    n = int(np.sqrt(lambda_mean.shape[0]))
    lambda_mean = lambda_mean.numpy().reshape(n,n)

    plt.xlim(X.min(), X.max())
    plt.pcolormesh(x_mesh, y_mesh, lambda_mean)
    plt.plot(events[:,0], events[:,1],'ro', markersize = 1.0)
    plt.colorbar()
    plt.show()
    


if __name__ == "__main__":
    demo()






