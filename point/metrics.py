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

from typing import Iterable, Dict

from point.helper import get_process, method
from point.utils import check_random_state_instance
from point.reward import Reward

from point.point_process import PointsData

Variables = Iterable[tf.Variable]
PointSequences = Iterable[np.ndarray]
Kernel = tfk.PositiveSemidefiniteKernel


class Metrics():
    
    def __init__(self, method =method.RFF, batch_size = 500, n_components = 250, random_state = None):
        self._method = method
        self._batch_size = batch_size
        self._n_components = n_components
        self._random_state = random_state


    def negative_rewards(self, expert_data : PointsData, **kwargs):
        
        random_state = check_random_state_instance(self._random_state)
        m = get_process(method.RFF, self._n_components, random_state, **kwargs)

        kernel = tfk.ExponentiatedQuadratic(amplitude=None, length_scale= tf.constant(0.5, dtype=default_float()))
        learner_data = m.generate(verbose = False, n_warm_up = 10000, batch_size = self._batch_size)
        rewards_per_batch = Reward().rewards_per_batch(learner_data, expert_data.locs, kernel)

        loss = sum(rewards_per_batch) / self._batch_size # := minus reward term

        return loss
    

    def log_likelihood(self, expert_data : PointsData, **kwargs):
        
        random_state = check_random_state_instance(self._random_state)
        m = get_process(method.RFF, self._n_components, random_state, **kwargs)
        
        loss = 0
        batch_per_resampling = 10
        
        for i in range(self._batch_size):
            m.sample()
            
            for idx in expert_data.shuffled_index(batch_per_resampling) :
                loss += m.lrgp.maximum_log_likelihood_objective(expert_data[idx])
                
        loss = - 1 * loss / (batch_per_resampling * self._batch_size)
        
        return loss[0]
    
    
    
    

    
if __name__ == "__main__":
    
    variance = tf.Variable([8], dtype=default_float(), name='sig')
    length_scale = tf.Variable([0.5], dtype=default_float(), name='l')
    rng = np.random.RandomState()
    
    #m = get_process(method.RFF, 250, rng, variance = variance, length_scale = length_scale)
    
    directory = "D:\GitHub\point\data\data_rff"
    expert_seq = np.load(directory + "\data_synth_points.npy")
    expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)
    expert_sizes = np.load(directory + "\data_synth_sizes.npy", allow_pickle=True)

    expert_data = PointsData(expert_sizes, expert_seq, expert_space)

    #loss1 = Metrics(batch_size = 500).negative_rewards(expert_data, variance = variance, length_scale = length_scale )
    loss1 = Metrics(batch_size = 1000).log_likelihood(expert_data, variance = variance, length_scale = length_scale )
    
    
    