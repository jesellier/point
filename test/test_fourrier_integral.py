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

float_type = tf.dtypes.float64
rng = np.random.RandomState(40)
from point.low_rank_gp import LowRankApproxGP



def repmat(vec, n):
    M =  tf.reshape(tf.tile(vec, tf.constant([n])), [n, tf.shape(vec)[0]])
    return M

def transformMat(vec, n):
    #for a vector vec return two matrices M1 = {v_i + v_j}_{i,j}  M1 = {v_i - v_j}_{i,j}
    M = repmat(vec, n)
    return (tf.transpose(M) + M, tf.transpose(M)-M)


def integral_cos_mat(w, a, b = tf.constant(0.0, dtype=float_type)):
    ### w = vector of weights, a = time, b = vector of b
    
    n =  w.shape[0]
    (A, C) = transformMat(w[:, 0], n)
    (B, D) = transformMat(w[:, 1], n)
    C = tf.linalg.set_diag(C, tf.ones(n,  dtype=float_type)) 
    D = tf.linalg.set_diag(D, tf.ones(n,  dtype=float_type)) 
    
    if b.shape == [] :
        b1 = 2 * b
        b2 = tf.constant(0.0, dtype=float_type)
    else :
        (b1, b2) = transformMat(b, n)
        
    
    mat = (1 / (A * B)) * ( tf.cos(a * A + b1) + tf.cos(a * B + b1) - tf.cos(a * (A + B) + b1)  - tf.cos(b1))
    mat += (1 / (C * D)) * ( tf.cos(a * C + b2) + tf.cos(a * D + b2) - tf.cos(a * (C + D) + b2)  - tf.cos(b2))
    
    bdo = 2 * b
    diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 *a * w[:,1] + bdo) + tf.cos(2 *a * w[:,0] + bdo) - tf.cos(2 *a * ( w[:,0] +  w[:,1] ) + bdo)  - tf.cos(bdo)) \
            +  a**2
    mat = tf.linalg.set_diag(mat, diag) 

    return  0.5 * mat



def term_plus(w, a, b = tf.constant(0.0, dtype=float_type)):
    n =  w.shape[0]
    A,_ = transformMat(w[:, 0], n)
    B,_ = transformMat(w[:, 1], n)
    
    mat = (1 / (A * B)) * ( tf.cos(a * A + b) + tf.cos(a * B + b) - tf.cos(a * (A + B) + b)  - tf.cos(b))
    diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 *a * w[:,1] + b) + tf.cos(2 *a * w[:,0] + b) - tf.cos(2 *a * ( w[:,0] +  w[:,1] ) + b)  - tf.cos(b))
    mat = tf.linalg.set_diag(mat, diag) 

    return mat

def term_minus(w, a, b = tf.constant(0.0, dtype=float_type)):
    n =  w.shape[0]
    _, C = transformMat(w[:, 0], n)
    _, D = transformMat(w[:, 1], n)
    
    C = tf.linalg.set_diag(C, tf.ones(n,  dtype=float_type)) 
    D = tf.linalg.set_diag(D, tf.ones(n,  dtype=float_type)) 
    
    mat =  (1 / (C * D))  * (tf.cos(a * C + b) + tf.cos(a * D + b) - tf.cos(a * (C + D) + b)  - tf.cos(b))
    mat = tf.linalg.set_diag(mat, tf.ones(n,  dtype=float_type) * tf.cos(b) * a**2) 
     
    return mat
 


class TestIntegralPart1(unittest.TestCase):

    #TESTS a = 1 ;   term = int [cos (w_i + w_j)^T x + 2] = int cos(x1 + x2 + 2)
    def test_1(self):
        a = 1.0
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=float_type, name='w')
        tmp = term_plus(w, a).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( 4 * np.cos(1) * np.sin(0.5)**2, val, places=7)
        
    # term = int cos(x1 + x2 + 2)
    def test_2(self):
        a = 1.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.0, 1.0],[1.0, 0.0]], dtype=float_type, name='w')
        tmp = term_plus(w, a, b).numpy()
        val = tmp[0,1]
        self.assertAlmostEqual(4 * np.cos(3) * np.sin(0.5)**2, val, places=7)
        
    def test_3(self):
        a = 1.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.0, 0.0],[1.0, 1.0]], dtype=float_type, name='w')
        tmp = term_plus(w, a, b).numpy()
        val = tmp[0,1]
        diag = tmp[1,1]
        self.assertAlmostEqual(4 * np.cos(3) * np.sin(0.5)**2, val, places=7)
        self.assertAlmostEqual( np.cos(4) * np.sin(1)**2, diag, places=7)

    # TEST a = 5 ;  int cos(x1 + x2 + 2 )
    def test_4(self):
        a = 5.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=float_type, name='w')
        tmp = term_plus(w, a, b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(4 * np.cos(7) * np.sin(5/2)**2, val, places=7)
        
    # TEST a = 1 ;  int cos(x1 + 2 * x2 + 2)
    def test_5(self):
        a = 1.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.5, 1],[0.5, 1]], dtype=float_type, name='w')
        tmp = term_plus(w, a, b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( (np.sin(1)*(np.sin(4)-np.sin(3))), val, places=7) 
       
    # TEST a = 5  ;  int cos(x1 + 2 * x2 )
    def test_6(self):
        a = 5.0
        w = tf.constant([[0.5, 1],[0.5, 1]], dtype=float_type, name='w')
        tmp = term_plus(w, a).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( ( 2 * np.cos(5) -1) * np.sin(5)**2, val, places=7)
        
        
class TestIntegralPart2(unittest.TestCase):

    # TEST second term when nul
    def test_1(self):
        a = 1.0
        w = tf.constant([[0.0, 0.0],[0.0, 0.0]], dtype=float_type, name='w')
        b = tf.constant(2.0, dtype=float_type)
        val = term_minus(w, a, b).numpy()

        self.assertAlmostEqual( np.cos(b.numpy()) * a ** 2, val[1,1], places=7)
        
    # test 
    def test_2(self):
        a = 1.0
        w = tf.constant([[2, 2],[1, 1]], dtype=float_type, name='w')
        val = term_minus(w, a).numpy()

        self.assertAlmostEqual( 4* np.cos(1) * np.sin(1/2)**2, val[0,1], places=7)
        self.assertAlmostEqual( 1.0, val[1,1], places=7)
    
    # add drift
    def test_3(self):
        
        a = 1.0
        w = tf.constant([[2, 2],[1.0, 1.0]], dtype=float_type, name='w')
        b = tf.constant(2.0, dtype=float_type)
        val = term_minus(w, a, b).numpy()

        self.assertAlmostEqual( 4* np.cos(3) * np.sin(1/2)**2, val[0,1], places=7)
        
        
        
        
class TestFullIntegral(unittest.TestCase):
    
    def setUp(self):
        self.gp = LowRankApproxGP(n_components = 2, random_state = rng)
        self.gp.fit(tf.ones(2, dtype=float_type))
        #mat3 = gp.__integral_mat(self, T = 1.0):
            

    
    #TESTS a = 1 ; int cos(b)cos(x1 + x2 + b)
    def test_1(self):
        a = 1.0
        w = tf.constant([[0.0, 0.0],[1.0, 1.0]], dtype=float_type, name='w')
        b = tf.constant(2.0, dtype=float_type)
        tmp1 = term_plus(w, a, 2*b).numpy()
        tmp2 = term_minus(w, a).numpy()
        
        mat1 = 0.5 * (tmp1 + tmp2 )
        mat2 =  integral_cos_mat(w, a, b).numpy()
        
        
        R = self.gp.randomFourier_.n_components
        self.gp.randomFourier_.random_weights_ = tf.transpose(w)
        self.gp.randomFourier_.random_offset_ = b
        mat3 = R * 0.5 * self.gp._LowRankApproxGP__integral_mat(T = a).numpy()
        
        #must process the Nan number in [0][0]
        mat1[np.isnan(mat1)] = np.inf
        mat2[np.isnan(mat2)] = np.inf
        mat3[np.isnan(mat3)] = np.inf

        self.assertTrue((mat2 == mat1).all())
        self.assertTrue((mat3 == mat2).all())
        self.assertAlmostEqual(4 * np.cos(3) * np.cos(2) * np.sin(0.5)**2, mat2[1,0], places=7)


    #TESTS a = 1, b = 0 ; int cos(0.1 * x1 + 0.2 * x2 )cos( 1.0 * x1 +  -2.0 * x2)
    def test_2(self):
        a = 1.0
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=float_type, name='w')
        tmp1 = term_plus(w, a).numpy()
        tmp2 = term_minus(w, a).numpy()
        
        mat1 = 0.5 * (tmp1 + tmp2 )
        mat2 =  integral_cos_mat(w, a).numpy()
        
        R = self.gp.randomFourier_.n_components
        self.gp.randomFourier_.random_weights_ = tf.transpose(w)
        self.gp.randomFourier_.random_offset_ = tf.constant(0.0, dtype=float_type)
        mat3 = R * 0.5 * self.gp._LowRankApproxGP__integral_mat(T = a).numpy()
  
        self.assertTrue((mat2 == mat1).all())
        self.assertTrue((mat3 == mat2).all())
        self.assertAlmostEqual(mat2[0,0], (1/4) * (-23 + 25 * np.cos(0.2) + 25 * np.cos(2/5) - 25 * np.cos(3/5)), places=7)
        self.assertAlmostEqual(mat2[1,1], (1/16) * (9 - np.cos(4)) , places=7)
        self.assertAlmostEqual(mat2[1,0], 0.7002116510508248 , places=7)
        
        
    def test_3(self):
        #TESTS a = 1, b = 0 ; int cos(0.1 * x1 + 0.2 * x2 + 2.0 )cos( 1.0 * x1 +  -2.0 * x2 + 3.0)
        a = 1.0
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=float_type, name='w')
        b = tf.constant([2.0, 3.0], dtype=float_type, name='b')
        mat1 =  integral_cos_mat(w, a, b).numpy()
        
        self.gp.randomFourier_.random_weights_ = tf.transpose(w)
        self.gp.randomFourier_.random_offset_ = b
        mat2 = self.gp._LowRankApproxGP__integral_mat(T = a).numpy()
        
        self.assertTrue((mat2 == mat1).all())
        self.assertAlmostEqual(mat1[0,0], 0.3012653529971747, places=7)
        self.assertAlmostEqual(mat1[1,1], 0.5542608460089069 , places=7)
        self.assertAlmostEqual(mat1[1,0], 0.3420353429585846 , places=7)


if __name__ == '__main__':
    unittest.main()








