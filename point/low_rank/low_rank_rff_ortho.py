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

    

        




    
    
    


        
        
        
        









    