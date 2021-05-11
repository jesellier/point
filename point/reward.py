# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:35:39 2021

@author: jesel
"""

import tensorflow as tf


class Reward():

    @staticmethod
    def __concenat_vector(vec):
        cont_vec =  tf.unstack(vec, axis = 0)  
        cont_vec =  tf.concat(cont_vec, axis=0)  
        cont_vec = tf.boolean_mask(cont_vec, (cont_vec[:,0] != 0))
        return cont_vec
    
    @classmethod
    def rewards(cls, learner_l, expert_l, kernel):
        
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