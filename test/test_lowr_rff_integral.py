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

import unittest

from gpflow.config import default_float

from point.helper import get_lrgp, method
from point.low_ranks.low_rank_rff import LowRankRFF
from point.point_process import Space

rng = np.random.RandomState(40)



def repmat(vec, n):
    M =  tf.reshape(tf.tile(vec, tf.constant([n])), [n, tf.shape(vec)[0]])
    return M

def transformMat(vec, n):
    #for a vector vec return two matrices M1 = {v_i + v_j}_{i,j}  M1 = {v_i - v_j}_{i,j}
    M = repmat(vec, n)
    return (tf.transpose(M) + M, tf.transpose(M)-M)


def integral_cos_mat(w, bounds, b = tf.constant(0.0, dtype=default_float())):
    ### w = vector of weights, a = time, b = vector of b
    
    n =  w.shape[0]
    (A, C) = transformMat(w[:, 0], n)
    (B, D) = transformMat(w[:, 1], n)
    C = tf.linalg.set_diag(C, tf.ones(n,  dtype=default_float())) 
    D = tf.linalg.set_diag(D, tf.ones(n,  dtype=default_float())) 
    
    up_bound = bounds[1]
    lo_bound = bounds[0]
 
    if b.shape == [] :
        b1 = 2 * b
        b2 = tf.constant(0.0, dtype=default_float())
    else :
        (b1, b2) = transformMat(b, n)

    mat = (1 / (A * B)) * ( tf.cos(up_bound*A + lo_bound*B + b1) + tf.cos(lo_bound*A + up_bound*B + b1)  \
                               - tf.cos(up_bound*(A + B) + b1)  - tf.cos(lo_bound*(A + B) + b1))
    mat += (1 / (C * D)) * (tf.cos(up_bound*C + lo_bound*D + b2) + tf.cos(lo_bound*C + up_bound*D + b2)  \
                                - tf.cos(up_bound*(C + D) + b2)  - tf.cos(lo_bound*(C + D) + b2 ))

    bdo = 2 * b
    diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (up_bound*w[:,0] + lo_bound* w[:,1]) + bdo) + tf.cos(2 * (lo_bound*w[:,0] + up_bound* w[:,1]) + bdo) \
                                              - tf.cos(2 *up_bound* (w[:,0]+ w[:,1]) + bdo) - tf.cos(2 *lo_bound*(w[:,0]+ w[:,1]) + bdo) ) \
                                              +  (up_bound - lo_bound)**2
    mat = tf.linalg.set_diag(mat, diag) 

    return  0.5 * mat


def term_plus(w, bounds, b = tf.constant(0.0, dtype=default_float())):
    n =  w.shape[0]
    A,_ = transformMat(w[:, 0], n)
    B,_ = transformMat(w[:, 1], n)
    
    up_bound = bounds[1]
    lo_bound = bounds[0]
 
    mat = (1 / (A * B)) * ( tf.cos(up_bound*A + lo_bound*B + b) + tf.cos(lo_bound*A + up_bound*B + b)  \
                               - tf.cos(up_bound*(A + B) + b)  - tf.cos(lo_bound*(A + B) + b))

    diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (up_bound*w[:,0] + lo_bound* w[:,1]) + b) + tf.cos(2 * (lo_bound*w[:,0] + up_bound* w[:,1]) + b) \
                                              - tf.cos(2 *up_bound* (w[:,0]+ w[:,1]) + b) - tf.cos(2 *lo_bound*(w[:,0]+ w[:,1]) + b) ) \

    mat = tf.linalg.set_diag(mat, diag) 

    return mat


def term_minus(w, bounds, b = tf.constant(0.0, dtype=default_float())):
    n =  w.shape[0]
    _, C = transformMat(w[:, 0], n)
    _, D = transformMat(w[:, 1], n)
    
    up_bound = bounds[1]
    lo_bound = bounds[0]
    
    C = tf.linalg.set_diag(C, tf.ones(n,  dtype=default_float())) 
    D = tf.linalg.set_diag(D, tf.ones(n,  dtype=default_float())) 

    mat = (1 / (C * D)) * (tf.cos(up_bound*C + lo_bound*D + b) + tf.cos(lo_bound*C + up_bound*D + b)  \
                                - tf.cos(up_bound*(C + D) + b)  - tf.cos(lo_bound*(C + D) + b ))
    mat = tf.linalg.set_diag(mat, tf.ones(n,  dtype=default_float()) * tf.cos(b) *  (up_bound - lo_bound)**2) 
     
    return mat
 


class TestIntegralPart1(unittest.TestCase):

    def test_1(self):
        #TESTS : bound = [0,1];  term = int [cos (w_i + w_j)^T x] = int cos(x1 + x2)
        bounds = [0, 1.0]
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_plus(w, bounds).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( 4 * np.cos(1) * np.sin(0.5)**2, val, places=7)
        
    
    def test_2(self):
        # Add drift : term = int [cos (w_i + w_j)^T x + 2] = int cos(x1 + x2 + 2)
        bounds = [0, 1.0]
        b = tf.constant(2.0, dtype=default_float())
        w = tf.constant([[0.0, 1.0],[1.0, 0.0]], dtype=default_float(), name='w')
        tmp = term_plus(w, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertAlmostEqual(4 * np.cos(3) * np.sin(0.5)**2, val, places=7)
        
    
    def test_3(self):
        # New term = int cos(2 *x2 + 2)
        bounds = [0, 1.0]
        b = tf.constant(2.0, dtype=default_float())
        w = tf.constant([[0.0, 0.0],[1.0, 1.0]], dtype=default_float(), name='w')
        tmp = term_plus(w, bounds, b=b).numpy()
        val = tmp[0,1]
        diag = tmp[1,1]
        self.assertAlmostEqual(4 * np.cos(3) * np.sin(0.5)**2, val, places=7)
        self.assertAlmostEqual( np.cos(4) * np.sin(1)**2, diag, places=7)


    def test_4(self):
        # Add non unit bound : [0,5];   int cos(x1 + x2 + 2 )
        bounds = [0, 5.0]
        b = tf.constant(2.0, dtype=default_float())
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_plus(w, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(4 * np.cos(7) * np.sin(5/2)**2, val, places=7)
        
    def test_5(self):
        # Add negative unit bound : [-1,1];   int cos(x1 + x2 + 2 )
        bounds = [-1, 1]
        b = tf.constant(2.0, dtype=default_float())
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_plus(w, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(4 * np.cos(2) * np.sin(1)**2, val, places=7)


    def test_7(self):
        # More test for completeness : [0,5];  int cos(x1 + 2*x2 )
        bounds = [0,5]
        w = tf.constant([[0.5, 1],[0.5, 1]], dtype=default_float(), name='w')
        tmp = term_plus(w, bounds).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( ( 2 * np.cos(5) -1) * np.sin(5)**2, val, places=7)
        
            
    def test_8(self):
        # More test for completeness : [0,5];  int cos(x1 + 2*x2 + 2)
        bounds = [0,5]
        b = tf.constant(2.0, dtype=default_float())
        w = tf.constant([[0.5, 1],[0.5, 1]], dtype=default_float(), name='w')
        tmp = term_plus(w, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( (np.sin(5)*(np.sin(12)-np.sin(7))), val, places=7) 

        
        
class TestIntegralPart2(unittest.TestCase):

    def test_1(self):
        # TEST second term when nul
        bounds = [0,1]
        w = tf.constant([[0.0, 0.0],[0.0, 0.0]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        val = term_minus(w = w, bounds = bounds, b = b).numpy()
        self.assertAlmostEqual( np.cos(b.numpy()) * bounds[1]** 2, val[1,1], places=7)
        
        
    def test_2(self):
        # TEST : term = int cos(x1 + x2)
        bounds = [0,1]
        w = tf.constant([[2, 2],[1, 1]], dtype=default_float(), name='w')
        val = term_minus(w = w, bounds = bounds).numpy()

        self.assertAlmostEqual( 4* np.cos(1) * np.sin(1/2)**2, val[0,1], places=7)
        self.assertAlmostEqual( 1.0, val[1,1], places=7)
    
    
    def test_3(self):
        # TEST add drift : term = int cos(x1 + x2 + 2)
        bounds = [0,1]
        w = tf.constant([[2, 2],[1, 1]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        val = term_minus(w = w, bounds = bounds, b = b).numpy()

        self.assertAlmostEqual( 4* np.cos(3) * np.sin(1/2)**2, val[0,1], places=7)
        
    def test_4(self):
        # TEST add negative bound
        bounds = [-1,1]
        w = tf.constant([[2, 2],[1, 1]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        val = term_minus(w = w, bounds = bounds, b = b).numpy()

        self.assertAlmostEqual( 4* np.cos(2) * np.sin(1)**2, val[0,1], places=7)
        
        
        
        
class TestFullIntegral(unittest.TestCase):
    
    def compMat(self, m1, m2):
        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                self.assertAlmostEqual( m1[i,j], m2[i,j] , places=7)
                
    
    def setUp(self):
        variance = tf.Variable(1, dtype=default_float(), name='sig')
        length_scale = tf.Variable([1, 1], dtype=default_float(), name='l')
        space =Space([0,1])
        
        self.gp = get_lrgp(method = method.RFF, variance = variance, length_scale = length_scale, space = space,
                        n_components = 2, random_state = rng)


    #TESTS bounds = [0,1]; diag_term = int cos(b)cos(x1 + x2 + b)
    def test_1(self):
        bounds = [0,1]
        w = tf.constant([[0.0, 0.0],[1.0, 1.0]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        tmp1 = term_plus(w=w, bounds = bounds,b=2*b).numpy()
        tmp2 = term_minus(w=w, bounds = bounds).numpy()
        
        mat1 = 0.5 * (tmp1 + tmp2)
        mat2 =  integral_cos_mat(w = w, bounds = bounds, b = b).numpy()

        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(w)
        self.gp._random_offset = b
        mat3 = R * 0.5 * self.gp._LowRankRFF__integral_mat(bounds).numpy()
        
        #must process the Nan number in [0][0]
        mat1[np.isnan(mat1)] = np.inf
        mat2[np.isnan(mat2)] = np.inf
        mat3[np.isnan(mat3)] = np.inf

        self.compMat(mat1, mat2)
        self.compMat(mat2, mat3)
        self.assertAlmostEqual(4 * np.cos(3) * np.cos(2) * np.sin(0.5)**2, mat3[1,0], places=7)


    def test_2(self):
        #TESTS bounds = [0,1] ; diag_term = int cos(0.1 * x1 + 0.2 * x2 )cos( 1.0 * x1 +  -2.0 * x2)
        bounds = [0,1]
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        tmp1 = term_plus(w, bounds).numpy()
        tmp2 = term_minus(w, bounds).numpy()
        
        mat1 = 0.5 * (tmp1 + tmp2 )
        mat2 =  integral_cos_mat(w, bounds).numpy()
        
        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(w)
        self.gp._random_offset = tf.constant(0.0, dtype=default_float())
        mat3 = R * 0.5 * self.gp._LowRankRFF__integral_mat(bounds).numpy()
  
        self.compMat(mat1, mat2)
        self.compMat(mat2, mat3)
        self.assertAlmostEqual(mat3[0,0], (1/4) * (-23 + 25 * np.cos(0.2) + 25 * np.cos(2/5) - 25 * np.cos(3/5)), places=7)
        self.assertAlmostEqual(mat3[1,1], (1/16) * (9 - np.cos(4)) , places=7)
        self.assertAlmostEqual(mat3[1,0], 0.7002116510508248 , places=7)
        
    
    def test_3(self):
        #TESTS Add drift : diag_term = int cos(0.1 * x1 + 0.2 * x2 + 2)cos( 1.0 * x1 +  -2.0 * x2 + 1)
        bounds = [0,1]
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        b = tf.constant([2, 1], dtype=default_float(), name='w')

        mat1 =  integral_cos_mat(w, bounds, b=b).numpy()
    
        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(w)
        self.gp._random_offset = b
        mat2 = R * 0.5 * self.gp._LowRankRFF__integral_mat(bounds).numpy()

        self.compMat(mat1, mat2)
        self.assertAlmostEqual(mat2[0,0], 0.3012653529971747, places=7)
        self.assertAlmostEqual(mat2[1,1], 0.6033527263039757, places=7)
        self.assertAlmostEqual(mat2[1,0], -0.39557712896935127 , places=7)
        
        
    def test_4(self):
        #TESTS neg bound : diag_term = int cos(0.1 * x1 + 0.2 * x2 + 2)cos( 1.0 * x1 +  -2.0 * x2 + 1)
        bounds = [-1,1]
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        b = tf.constant([2, 1], dtype=default_float(), name='w')

        mat1 =  integral_cos_mat(w, bounds, b=b).numpy()
    
        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(w)
        self.gp._random_offset = b
        self.gp.space = Space([-1,1])
        
        mat2 = R * 0.5 * self.gp._LowRankRFF__integral_mat(bounds).numpy()

        self.compMat(mat1, mat2)
        self.assertAlmostEqual(mat2[0,0], 0.7357636641212484, places=7)
        self.assertAlmostEqual(mat2[1,1], 2.0715937521130385, places=7)
        self.assertAlmostEqual(mat2[1,0], -0.5222545782912589, places=7)
 
    def test_5(self):
        #TESTS a = 1, b = 0 ; int cos(0.1 * x1 + 0.2 * x2 + 2.0 )cos( 1.0 * x1 +  -2.0 * x2 + 3.0)
        bounds = [0,1]
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        b = tf.constant([2.0, 3.0], dtype=default_float(), name='b')
        mat1 =  integral_cos_mat(w=w, bounds = bounds, b=b).numpy()
        
        self.gp._random_weights = tf.transpose(w)
        self.gp._random_offset = b
        mat2 = self.gp._LowRankRFF__integral_mat(bounds).numpy()
        
        self.compMat(mat1, mat2)
        self.assertAlmostEqual(mat2[0,0], 0.3012653529971747, places=7)
        self.assertAlmostEqual(mat2[1,1], 0.5542608460089069 , places=7)
        self.assertAlmostEqual(mat2[1,0], 0.3420353429585846 , places=7)


if __name__ == '__main__':
    unittest.main()








