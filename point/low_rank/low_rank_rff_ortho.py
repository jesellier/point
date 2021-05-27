# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:20:59 2021
@author: jesel
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float
import gpflow.kernels as gfk 


from point.low_rank.low_rank_rff import LowRankRFF
from point.utils import check_random_state_instance
from point.misc import Space

from scipy.linalg import qr_multiply
from scipy.stats import chi




class LowRankRFFOrthogonal(LowRankRFF):
    
    def __init__(self, kernel, beta0 = 1e-10, space = Space(), n_components = 1000, random_state = None):
       
        super().__init__(kernel, beta0, space, n_components, random_state)
        self.n_stacks = int(np.ceil(self.n_components/self.n_features))
        self.n_components = self.n_stacks * self.n_features
        

    def sample(self):
        random_state = check_random_state_instance(self._random_state)
        size = (self.n_features, self.n_features)

        G = []
        for _ in range(self.n_stacks):
             W = random_state.randn(*size)
             S = np.diag(chi.rvs(df=self.n_features, size=self.n_features, random_state=random_state))
             SQ, _ = qr_multiply(W, S)
             G += [SQ]

        self._G = tf.constant(np.vstack(G).T, dtype= default_float(), name='G')
        self._random_offset = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype= default_float(), name='b')
        self._latent = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=default_float(), name='latent')
        





if __name__ == '__main__':
    rng = np.random.RandomState(20)

    beta0 = tf.Variable([0.5], dtype=default_float(), name='beta0')
    variance = tf.Variable([5], dtype=default_float(), name='sig')
    length_scale = tf.Variable([0.2,0.2], dtype=default_float(), name='lenghtscale')
    kernel = gfk.SquaredExponential(variance= variance , lengthscales= length_scale)

    lrgp = LowRankRFFOrthogonal(kernel, beta0 = beta0, n_components = 250, random_state = rng).fit()
    X = tf.constant(rng.normal(size = [100, 2]), dtype=default_float(), name='X')

    K = kernel(X).numpy()
    K2 = lrgp(X).numpy()

    #print(lrgp.maximum_log_likelihood_objective(X))
    #print(lrgp.func(X))

    #lrgp.plot_kernel()
    #lrgp.plot_surface()

    

        




    
    
    


        
        
        
        









    