# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64
rng = np.random.RandomState(40)

import unittest

from point.low_rank_gp import  LowRankApproxGP
from point.utils import transformMat




def integral_mat_recalc(w, b, R, lplus = 1.0, lminus = 0):

    w = tf.transpose(w)
    (A, C) = transformMat(w[:, 0], R)
    (B, D) = transformMat(w[:, 1], R)
    C = tf.linalg.set_diag(C, tf.ones(R,  dtype=float_type)) 
    D = tf.linalg.set_diag(D, tf.ones(R,  dtype=float_type)) 

    if b.shape == [] :
        b1 = 2 * b
        b2 = tf.constant(0.0, dtype=float_type)
    else :
        (b1, b2) = transformMat(b, R)

    mat = (1 / (A * B)) * ( tf.cos(lplus*A + lminus*B + b1) + tf.cos(lminus*A + lplus*B + b1)  \
                           - tf.cos(lplus*(A + B) + b1)  - tf.cos(lminus*(A + B) + b1))
    mat += (1 / (C * D)) * (tf.cos(lplus*C + lminus*D + b2) + tf.cos(lminus*C + lplus*D + b2)  \
                            - tf.cos(lplus*(C + D) + b2)  - tf.cos(lminus*(C + D) + b2 ))
    
    bdo = 2 * b
    diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (lplus*w[:,0] + lminus* w[:,1]) + bdo) + tf.cos(2 * (lminus*w[:,0] + lplus* w[:,1]) + bdo) \
                                          - tf.cos(2 *lplus* (w[:,0]+ w[:,1]) + bdo) - tf.cos(2 *lminus*(w[:,0]+ w[:,1]) + bdo) ) \
                                          +  (lplus - lminus)**2
    mat = tf.linalg.set_diag(mat, diag) 

    return  mat / R



class Test_Gradient(unittest.TestCase):
    
    
    def setUp(self):
        
        self.n_components = 2
        self.variance = tf.Variable(2.0, dtype=float_type, name='sig')
        
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        self.gp = LowRankApproxGP(n_components = self.n_components,  random_state = rng)
        self.X = tf.constant(rng.normal(size = [10, 2]), dtype=float_type, name='X')

        
    def test_variance_grad(self):
        gp = self.gp
        variance = self.variance
        length_scale = self.length_scale
        X = self.X
        
        gp.fit(length_scale, variance)
        likelihood, grad = gp.likelihood_grad(X, lplus = 1.0, lminus = 0.0)
        grad_variance = grad[0].numpy()
        true_value = (likelihood/variance).numpy()
        self.assertAlmostEqual(true_value , grad_variance, places=7)
        
        
    def test_grad_process(self):
        gp = self.gp
        variance = self.variance
        length_scale = self.length_scale

        gp.fit(length_scale, variance)
        out, grad = gp.integral_grad(lplus = 1.0, lminus = 0.0)
        grad = grad[1]

        R = gp.n_components
        b = gp.randomFourier_.random_offset_
        z = gp.randomFourier_.z_
        beta = gp.beta_
        
        print(grad)
        
        with tf.GradientTape() as tape:  
            gamma = 1 / (2 * length_scale **2 )       
            gamma = tf.math.sqrt(2 * gamma)
            w = tf.linalg.diag(gamma)  @ z  
            mat = integral_mat_recalc(w, b, R, lplus = 1.0, lminus = 0)
            out2 = variance * tf.transpose(beta) @ mat @ beta
        grad2 = tape.gradient(out2, length_scale) 

        self.assertAlmostEqual(out.numpy(), out2.numpy(), places=7)
        for g in zip(grad.numpy(), grad2.numpy()):
            self.assertAlmostEqual(g[0], g[1], places=7)
     

        

        
        
if __name__ == '__main__':
    unittest.main()


 