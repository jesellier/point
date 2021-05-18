# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:35:39 2021

@author: jesel
"""

from typing import Iterable

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import numpy as np

from point.point_process import PointsData

Variables = Iterable[tf.Variables]
PointSequences = Iterable[np.ndarray]  # deprecated
Kernel = tfk.PositiveSemidefiniteKernel


class Reward():

    @staticmethod
    def __concenat_vector(vec : PointSequences) -> Variables :
        cont_vec =  tf.unstack(vec, axis = 0)  
        cont_vec =  tf.concat(cont_vec, axis=0)  
        cont_vec = tf.boolean_mask(cont_vec, (cont_vec[:,0] != 0))
        return cont_vec
    
    @classmethod
    def rewards_per_points(cls, learner_l : PointSequences, expert_l : PointSequences, kernel: Kernel):
        
        n_learner_batches = len(learner_l)
        n_expert_batches = len(expert_l)
    
        learner_l = cls.__concenat_vector(learner_l)
        expert_l = cls.__concenat_vector(expert_l) 
        
        mat_el = kernel.matrix(expert_l, learner_l)
        mat_ll = kernel.matrix(learner_l, learner_l)
    
        sum_el = tf.math.reduce_sum(mat_el , axis = 0)
        sum_ll = tf.math.reduce_sum(mat_ll , axis = 0)
        rewards = sum_el / n_expert_batches  - sum_ll / n_learner_batches
        
        return rewards
    
    @classmethod
    def rewards_per_batch(cls, learner_data : PointsData, expert_l : PointSequences, kernel : Kernel):

        rewards_per_points = cls.rewards_per_points(learner_data.locs, expert_l, kernel)
        rewards_per_batch =[ tf.math.reduce_sum(s) for s in tf.split(rewards_per_points, learner_data.sizes )]
        rewards_per_batch = tf.stack(rewards_per_batch)

        return rewards_per_batch
