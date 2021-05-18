# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:35:39 2021

@author: jesel
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from gpflow.config import default_float

from typing import Iterable

from point.helper import get_process, method
from point.utils import check_random_state_instance
from point.reward import Reward

from point.point_process import PointsData

Variables = Iterable[tf.Variables]
PointSequences = Iterable[np.ndarray]
Kernel = tfk.PositiveSemidefiniteKernel


class Metrics():
    
    def __init__(self, method =method.RFF, batch_size = 500, n_components = 250, random_state = None):
        self._method = method
        self._batch_size = batch_size
        self._n_components = n_components
        self._random_state = random_state


    @classmethod
    def negative_rewards(self, expert_data : PointsData, variables : dict[str, tf.Variables]):
        
        random_state = check_random_state_instance(self._random_state)
        m = get_process(method.RFF, self._n_components, random_state, variables)

        kernel = tfk.ExponentiatedQuadratic(amplitude=None, length_scale= tf.constant(0.5, dtype=default_float()))
        learner_data = m.generate(verbose = False, n_warm_up = 10000, batch_size = self.batch_size)
        rewards_per_batch = Reward().rewards_per_batch(learner_data, expert_data.locs, kernel)

        loss = sum(rewards_per_batch) / self.batch_size # := minus reward term

        return loss
    
    
    @classmethod
    def likelihood(self, expert_data : PointsData, variables : dict[str, tf.Variables]):
        
        random_state = check_random_state_instance(self._random_state)
        m = get_process(method.RFF, self._n_components, random_state, variables)
        
        loss = 0
        for _ in range(self.batch_size):
            m.sample()
            for e in expert_data :
                loss += m.lrgp.maximum_log_likelihood_objective(e)
        
        return loss / self.batch_size