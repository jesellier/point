# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

rng = np.random.RandomState(40)

import unittest

def concenat_vector(vec):
    cont_vec =  tf.unstack(vec, axis = 0)  
    cont_vec =  tf.concat(cont_vec, axis=0)  
    cont_vec = tf.boolean_mask(cont_vec, (cont_vec[:,0] != 0))
    return cont_vec


def reward_vector(learner_l, expert_l, kernel):
    
    n_learner_batches = len(learner_l)
    n_expert_batches = len(expert_l)

    learner_l = concenat_vector(learner_l)
    expert_l = concenat_vector(expert_l) 
    
    mat_el = kernel.matrix(expert_l, learner_l)
    mat_ll = kernel.matrix(learner_l, learner_l)

    sum_el = tf.math.reduce_sum(mat_el , axis = 0)
    sum_ll = tf.math.reduce_sum(mat_ll , axis = 0)
    reward = sum_el / n_expert_batches  - sum_ll / n_learner_batches
    
    return (reward, sum_el, sum_ll)

            


class Test_Reward(unittest.TestCase):
    
    
    def convert_to_list(learner):
        list_p = []
        for batch in learner :
            for e in batch :
                list_p.append(e)
        return list_p
    
    
    def compute_sum(learner, point, kernel):
            s = 0
            for batch in learner :
                for e in batch :
                    s +=  kernel.apply(e, point)     
            return s
        
    
    def setUp(self):
        
        n_batch = 2
        size = 10

        self.learner_l = [rng.normal(size = [size, 2]) for _ in range(n_batch)]
        self.expert_l =  [rng.normal(size = [size, 2]) for _ in range(n_batch)]
        
        self.kernel = tfk.ExponentiatedQuadratic(
            amplitude=None, 
            length_scale= tf.constant(0.5,  dtype=float_type),
            name='ExponentiatedQuadratic')
        
        
    def test_concat(self):
        
        learner = concenat_vector(self.learner_l)
        bench_learner = Test_Reward.convert_to_list(self.learner_l)
        
        for i in range(len(learner)):
            self.assertAlmostEqual(learner[i][0], bench_learner[i][0], places=7)
            self.assertAlmostEqual(learner[i][1], bench_learner[i][1], places=7)
                    
        
        
    def test_sum(self):
        
        learner_l = self.learner_l
        expert_l  = self.expert_l 
        kernel = self.kernel

        reward, sel, sll = reward_vector(learner_l, expert_l, kernel)
        points = Test_Reward.convert_to_list(self.learner_l)
        
        
        for i in range(len(points)):
            recalc = Test_Reward.compute_sum(learner_l, points[i], kernel)
            self.assertAlmostEqual(sll[i].numpy() , recalc.numpy() , places=7)
            
        for i in range(len(points)):
            recalc = Test_Reward.compute_sum(expert_l, points[i], kernel)
            self.assertAlmostEqual(sel[i].numpy() , recalc.numpy() , places=7)




if __name__ == '__main__':
    unittest.main()

 
    
    