import numpy as np
import arrow
import sys
import matplotlib.pyplot as plt

#import tensorflow as tf
#import tensorflow_probability as tfp
#tfd = tfp.distributions
#tfk = tfp.math.psd_kernels

#float_type = tf.dtypes.float64

from scipy.optimize import minimize #For optimizing
from point.utils import check_random_state_instance

from bayesoptim.utils import UtilityFunction
from bayesoptim.optimizer import  ModifiedBayesianOptimizer

from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


from mpl_toolkits.mplot3d import Axes3D  # Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt


rng = np.random.RandomState(10)
variance = 0.2
length_scale = [10,10]

kernel = kernel = variance * RBF(length_scale= length_scale)

gp = GaussianProcessRegressor(
        kernel= kernel,
        alpha=  1e-6,
        normalize_y= False,
        optimizer = True
        #random_state= rng
    )


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-1.0, 1.0, 0.05)
X, Y = np.meshgrid(x, y)

n = X.shape[0]**2
inputs = np.zeros((n,2))
inputs[:,0] = np.ravel(X)
inputs[:,1] = np.ravel(Y)

zs = np.array(gp.sample_y(inputs))
Z = zs.reshape(X.shape)

K = kernel(inputs)

ax.plot_surface(X, Y, Z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
    
    
    


        
    
    