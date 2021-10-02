import numpy as np
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt

import tensorflow as tf

from point.utils import check_random_state_instance
from point.model import InHomogeneousSpatialModel
from point.misc import Space

from gpflow.config import default_float




def build_data_400():
    dim = 2
    num = 200
    bound =  np.array([-10, 10 ]) 
    rng=np.random.RandomState(10)
    
    right = rng.uniform(np.array([0,-10]) , np.array([10,10]), size=(int(num/2), dim))
    top_left  = rng.uniform(np.array([-10,-0.0]) , np.array([0,10]), size=(int(num/8), dim))
    
    central_cluster = rng.uniform(low = -1, high = 1, size=(int(num/2), dim))
    events4 = rng.uniform(low = 6, high = 8, size=(int(num/2), dim))
    events5 = rng.uniform(np.array([5,-7.5]) , np.array([7.5,-2.5]), size=(int(num/2), dim))
    events = np.vstack((right, top_left , central_cluster, events4, events5))
    return events, Space(bound)


def build_data_1000():
    dim = 2
    num = 250
    bound = np.array([-15, 15 ]) 
    rng=np.random.RandomState(10)
    
    right = rng.uniform(np.array([0,-15]) , np.array([15,15]), size=(int(num), dim))
    top_left = rng.uniform(np.array([-15,-0.0]) , np.array([0,15]), size=(int(num/2), dim))
    noise = np.vstack((right, top_left))
    
    central_cluster1 = rng.uniform(low = -3, high = 1, size=(int(num/3), dim))
    central_cluster2 = rng.uniform(np.array([0,-10]) , np.array([3,-1]), size=(int(num/3), dim))
    central_cluster = np.vstack((central_cluster1, central_cluster2))
    
    right_cluster1 = rng.uniform(low = 4, high = 9, size=(int(num/2), dim))
    right_cluster2 = rng.uniform(low = 5, high = 8, size=(int(num/3), dim))
    right_cluster3  = rng.uniform(low = 6, high = 7, size=(int(num/4), dim))
    right_cluster = np.vstack((right_cluster1, right_cluster2, right_cluster3))
    
    left_cluster1 = rng.uniform(np.array([8,-15]) , np.array([15,-5]), size=(int(num), dim))
    left_cluster2 = rng.uniform(np.array([10,-14]) , np.array([13,-6]), size=(int(num/3), dim))
    left_cluster3 = rng.uniform(np.array([11,-12]) , np.array([12,-9]), size=(int(num/3), dim))
    left_cluster =  np.vstack((left_cluster1, left_cluster2, left_cluster3))
    
    events = np.vstack((noise, central_cluster, right_cluster, left_cluster))
    return events, Space(bound)


def lambda_synth_400():
    directory = "D:\GitHub\point\data\data_synth_400.csv"
    data = genfromtxt(directory, delimiter=',')
    lambda_sample = data[:,2]
    grid = data[:,0:2]
    
    lambda_sample = tf.expand_dims(tf.convert_to_tensor(lambda_sample, dtype=default_float()),1)
    bound = (int(grid.min()), int(math.ceil(grid.max())))
    return lambda_sample, grid, Space(bound)

def lambda_synth_1000():
    directory = "D:\GitHub\point\data\data_synth_1000_2.csv"
    data = genfromtxt(directory, delimiter=',')
    lambda_sample = data[:,2]
    grid = data[:,0:2]
    
    lambda_sample = tf.expand_dims(tf.convert_to_tensor(lambda_sample, dtype=default_float()),1)
    bound = (int(grid.min()), int(math.ceil(grid.max())))
    return lambda_sample, grid, Space(bound)


def get_synthetic_generative_model(lambda_sample = None, grid = None, random_state = None):
    
    if lambda_sample is None :
        lambda_sample, grid = lambda_synth_1000()

    random_state = check_random_state_instance(random_state)
    bound = (int(grid.min()), int(math.ceil(grid.max())))
    return InHomogeneousSpatialModel(grid, lambda_sample, bound, random_state) 

    
def print_grid(grid, lambda_sample, X = None):
    n = int(np.sqrt(lambda_sample.shape[0]))
    
    if tf.is_tensor(lambda_sample):
         lambda_sample = lambda_sample.numpy()
         
    lambda_matrix = lambda_sample.reshape(n,n)
    plt.xlim(grid.min(), grid.max())
    plt.pcolormesh(np.unique(grid[:,0]), np.unique(grid[:,1]), lambda_matrix, shading='auto')
    if X is not None :
        plt.plot(X[:,0], X[:,1],'ro', markersize = 0.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=8)
    plt.show()


def domain_grid(bound = (-10,10) , step = 0.1):
     x_mesh = y_mesh = np.arange(bound [0], bound [1] + step, step)
     X, Y = np.meshgrid(x_mesh, y_mesh)
     grid = np.zeros((X.shape[0]**2,2))
     grid[:,0] = np.ravel(X)
     grid[:,1] = np.ravel(Y)
     return grid, x_mesh, y_mesh
 

 
