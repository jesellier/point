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
from point.low_rank_rff import LowRankRFF



def repmat(vec, n):
    M =  tf.reshape(tf.tile(vec, tf.constant([n])), [n, tf.shape(vec)[0]])
    return M

def transformMat(vec, n):
    #for a vector vec return two matrices M1 = {v_i + v_j}_{i,j}  M1 = {v_i - v_j}_{i,j}
    M = repmat(vec, n)
    return (tf.transpose(M) + M, tf.transpose(M)-M)



def integral_cos_mat(w, lplus, lminus = 0, b = tf.constant(0.0, dtype=float_type)):
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

    mat = (1 / (A * B)) * ( tf.cos(lplus*A + lminus*B + b1) + tf.cos(lminus*A + lplus*B + b1)  \
                               - tf.cos(lplus*(A + B) + b1)  - tf.cos(lminus*(A + B) + b1))
    mat += (1 / (C * D)) * (tf.cos(lplus*C + lminus*D + b2) + tf.cos(lminus*C + lplus*D + b2)  \
                                - tf.cos(lplus*(C + D) + b2)  - tf.cos(lminus*(C + D) + b2 ))
        
    bdo = 2 * b
    diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (lplus*w[:,0] + lminus* w[:,1]) + bdo) + tf.cos(2 * (lminus*w[:,0] + lplus* w[:,1]) + bdo) \
                                              - tf.cos(2 *lplus* (w[:,0]+ w[:,1]) + bdo) - tf.cos(2 *lminus*(w[:,0]+ w[:,1]) + bdo) ) \
                                              +  (lplus - lminus)**2
    mat = tf.linalg.set_diag(mat, diag) 

    return  0.5 * mat



def term_plus(w, lplus, lminus = 0, b = tf.constant(0.0, dtype=float_type)):
    n =  w.shape[0]
    A,_ = transformMat(w[:, 0], n)
    B,_ = transformMat(w[:, 1], n)
 
    mat = (1 / (A * B)) * ( tf.cos(lplus*A + lminus*B + b) + tf.cos(lminus*A + lplus*B + b)  \
                               - tf.cos(lplus*(A + B) + b)  - tf.cos(lminus*(A + B) + b))

    diag = (1 / (4 * w[:,0] * w[:,1])) * ( tf.cos(2 * (lplus*w[:,0] + lminus* w[:,1]) + b) + tf.cos(2 * (lminus*w[:,0] + lplus* w[:,1]) + b) \
                                              - tf.cos(2 *lplus* (w[:,0]+ w[:,1]) + b) - tf.cos(2 *lminus*(w[:,0]+ w[:,1]) + b) ) \

    mat = tf.linalg.set_diag(mat, diag) 

    return mat


def term_minus(w, lplus, lminus = 0, b = tf.constant(0.0, dtype=float_type)):
    n =  w.shape[0]
    _, C = transformMat(w[:, 0], n)
    _, D = transformMat(w[:, 1], n)
    
    C = tf.linalg.set_diag(C, tf.ones(n,  dtype=float_type)) 
    D = tf.linalg.set_diag(D, tf.ones(n,  dtype=float_type)) 

    mat = (1 / (C * D)) * (tf.cos(lplus*C + lminus*D + b) + tf.cos(lminus*C + lplus*D + b)  \
                                - tf.cos(lplus*(C + D) + b)  - tf.cos(lminus*(C + D) + b ))
    mat = tf.linalg.set_diag(mat, tf.ones(n,  dtype=float_type) * tf.cos(b) *  (lplus - lminus)**2) 
     
    return mat
 


class TestIntegralPart1(unittest.TestCase):

    def test_1(self):
        #TESTS : lplus = 1, lminus = 0;  term = int [cos (w_i + w_j)^T x] = int cos(x1 + x2)
        lplus = 1.0
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=float_type, name='w')
        tmp = term_plus(w, lplus).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( 4 * np.cos(1) * np.sin(0.5)**2, val, places=7)
        
    
    def test_2(self):
        # Add drift : term = int [cos (w_i + w_j)^T x + 2] = int cos(x1 + x2 + 2)
        lplus = 1.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.0, 1.0],[1.0, 0.0]], dtype=float_type, name='w')
        tmp = term_plus(w, lplus, b=b).numpy()
        val = tmp[0,1]
        self.assertAlmostEqual(4 * np.cos(3) * np.sin(0.5)**2, val, places=7)
        
    
    def test_3(self):
        # New term = int cos(2 *x2 + 2)
        lplus = 1.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.0, 0.0],[1.0, 1.0]], dtype=float_type, name='w')
        tmp = term_plus(w, lplus, b=b).numpy()
        val = tmp[0,1]
        diag = tmp[1,1]
        self.assertAlmostEqual(4 * np.cos(3) * np.sin(0.5)**2, val, places=7)
        self.assertAlmostEqual( np.cos(4) * np.sin(1)**2, diag, places=7)


    def test_4(self):
        # Add non unit bound : lplus = 5, lminus = 0;   int cos(x1 + x2 + 2 )
        lplus = 5.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=float_type, name='w')
        tmp = term_plus(w, lplus, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(4 * np.cos(7) * np.sin(5/2)**2, val, places=7)
        
    def test_5(self):
        # Add negative unit bound : lplus = 1, lminus = -1;   int cos(x1 + x2 + 2 )
        lplus = 1.0
        lminus = -1.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=float_type, name='w')
        tmp = term_plus(w, lplus, lminus, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(4 * np.cos(2) * np.sin(1)**2, val, places=7)


    def test_7(self):
        # More test for completeness : lplus = 5, lminus = 0;  int cos(x1 + 2*x2 )
        lplus = 5.0
        w = tf.constant([[0.5, 1],[0.5, 1]], dtype=float_type, name='w')
        tmp = term_plus(w, lplus).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( ( 2 * np.cos(5) -1) * np.sin(5)**2, val, places=7)
        
            
    def test_8(self):
        # More test for completeness : lplus = 5, lminus = 0;  int cos(x1 + 2*x2 + 2)
        lplus = 1.0
        b = tf.constant(2.0, dtype=float_type)
        w = tf.constant([[0.5, 1],[0.5, 1]], dtype=float_type, name='w')
        tmp = term_plus(w, lplus, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( (np.sin(1)*(np.sin(4)-np.sin(3))), val, places=7) 

        
        
class TestIntegralPart2(unittest.TestCase):

    def test_1(self):
        # TEST second term when nul
        lplus = 1.0
        w = tf.constant([[0.0, 0.0],[0.0, 0.0]], dtype=float_type, name='w')
        b = tf.constant(2.0, dtype=float_type)
        val = term_minus(w = w, lplus = lplus, b = b).numpy()
        self.assertAlmostEqual( np.cos(b.numpy()) * lplus ** 2, val[1,1], places=7)
        
        
    def test_2(self):
        # TEST : term = int cos(x1 + x2)
        lplus = 1.0
        w = tf.constant([[2, 2],[1, 1]], dtype=float_type, name='w')
        val = term_minus(w = w, lplus = lplus).numpy()

        self.assertAlmostEqual( 4* np.cos(1) * np.sin(1/2)**2, val[0,1], places=7)
        self.assertAlmostEqual( 1.0, val[1,1], places=7)
    
    
    def test_3(self):
        # TEST add drift : term = int cos(x1 + x2 + 2)
        lplus = 1.0
        w = tf.constant([[2, 2],[1, 1]], dtype=float_type, name='w')
        b = tf.constant(2.0, dtype=float_type)
        val = term_minus(w = w, lplus = lplus, b = b).numpy()

        self.assertAlmostEqual( 4* np.cos(3) * np.sin(1/2)**2, val[0,1], places=7)
        
    def test_4(self):
        # TEST add negative bound
        lplus = 1.0
        lminus = -1.0
        w = tf.constant([[2, 2],[1, 1]], dtype=float_type, name='w')
        b = tf.constant(2.0, dtype=float_type)
        val = term_minus(w = w, lplus = lplus, lminus = lminus, b = b).numpy()

        self.assertAlmostEqual( 4* np.cos(2) * np.sin(1)**2, val[0,1], places=7)
        
        
        
        
class TestFullIntegral(unittest.TestCase):
    
    def setUp(self):
        variance = tf.ones(1, dtype=float_type)
        lenghtscale = (tf.ones(2, dtype=float_type))
        self.gp = LowRankRFF(lenghtscale, variance, n_components = 2, random_state = rng)
        #self.trainable_variables = (tf.ones(2, dtype=float_type), tf.ones(1, dtype=float_type))
        self.gp.fit()  #dummy fitting


    #TESTS lplus = 1, lminus = 0; diag_term = int cos(b)cos(x1 + x2 + b)
    def test_1(self):
        lplus = 1.0
        w = tf.constant([[0.0, 0.0],[1.0, 1.0]], dtype=float_type, name='w')
        b = tf.constant(2.0, dtype=float_type)
        tmp1 = term_plus(w=w, lplus=lplus,b=2*b).numpy()
        tmp2 = term_minus(w=w, lplus=lplus).numpy()
        
        mat1 = 0.5 * (tmp1 + tmp2 )
        mat2 =  integral_cos_mat(w = w, lplus = lplus, b = b).numpy()

        R = self.gp.n_components
        self.gp.random_weights_ = tf.transpose(w)
        self.gp.random_offset_ = b
        mat3 = R * 0.5 * self.gp._LowRankRFF__integral_mat(lplus = lplus).numpy()
        
        #must process the Nan number in [0][0]
        mat1[np.isnan(mat1)] = np.inf
        mat2[np.isnan(mat2)] = np.inf
        mat3[np.isnan(mat3)] = np.inf

        self.assertTrue((mat2 == mat1).all())
        self.assertTrue((mat3 == mat2).all())
        self.assertAlmostEqual(4 * np.cos(3) * np.cos(2) * np.sin(0.5)**2, mat2[1,0], places=7)


    def test_2(self):
        #TESTS lplus = 1, lminus = 0 ; diag_term = int cos(0.1 * x1 + 0.2 * x2 )cos( 1.0 * x1 +  -2.0 * x2)
        lplus = 1.0
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=float_type, name='w')
        tmp1 = term_plus(w, lplus).numpy()
        tmp2 = term_minus(w, lplus).numpy()
        
        mat1 = 0.5 * (tmp1 + tmp2 )
        mat2 =  integral_cos_mat(w, lplus).numpy()
        
        R = self.gp.n_components
        self.gp.random_weights_ = tf.transpose(w)
        self.gp.random_offset_ = tf.constant(0.0, dtype=float_type)
        mat3 = R * 0.5 * self.gp._LowRankRFF__integral_mat(lplus = lplus).numpy()
  
        self.assertTrue((mat2 == mat1).all())
        self.assertTrue((mat3 == mat2).all())
        self.assertAlmostEqual(mat2[0,0], (1/4) * (-23 + 25 * np.cos(0.2) + 25 * np.cos(2/5) - 25 * np.cos(3/5)), places=7)
        self.assertAlmostEqual(mat2[1,1], (1/16) * (9 - np.cos(4)) , places=7)
        self.assertAlmostEqual(mat2[1,0], 0.7002116510508248 , places=7)
        
    
    def test_3(self):
        #TESTS Add drift : diag_term = int cos(0.1 * x1 + 0.2 * x2 + 2)cos( 1.0 * x1 +  -2.0 * x2 + 1)
        lplus = 1.0
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=float_type, name='w')
        b = tf.constant([2, 1], dtype=float_type, name='w')

        mat1 =  integral_cos_mat(w, lplus, b=b).numpy()
    
        R = self.gp.n_components
        self.gp.random_weights_ = tf.transpose(w)
        self.gp.random_offset_ = b
        mat2 = R * 0.5 * self.gp._LowRankRFF__integral_mat(lplus = lplus).numpy()

        self.assertTrue((mat2 == mat1).all())
        self.assertAlmostEqual(mat2[0,0], 0.3012653529971747, places=7)
        self.assertAlmostEqual(mat2[1,1], 0.6033527263039757, places=7)
        self.assertAlmostEqual(mat2[1,0], -0.39557712896935127 , places=7)
        
        
    def test_4(self):
        #TESTS neg bound : diag_term = int cos(0.1 * x1 + 0.2 * x2 + 2)cos( 1.0 * x1 +  -2.0 * x2 + 1)
        lplus = 1.0
        lminus = -1.0
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=float_type, name='w')
        b = tf.constant([2, 1], dtype=float_type, name='w')

        mat1 =  integral_cos_mat(w, lplus, lminus, b=b).numpy()
    
        R = self.gp.n_components
        self.gp.random_weights_ = tf.transpose(w)
        self.gp.random_offset_ = b
        mat2 = R * 0.5 * self.gp._LowRankRFF__integral_mat(lplus = lplus, lminus = lminus).numpy()

        self.assertTrue((mat2 == mat1).all())
        self.assertAlmostEqual(mat2[0,0], 0.7357636641212484, places=7)
        self.assertAlmostEqual(mat2[1,1], 2.0715937521130385, places=7)
        self.assertAlmostEqual(mat2[1,0], -0.5222545782912589, places=7)
 
    def test_5(self):
        #TESTS a = 1, b = 0 ; int cos(0.1 * x1 + 0.2 * x2 + 2.0 )cos( 1.0 * x1 +  -2.0 * x2 + 3.0)
        lplus = 1.0
        w = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=float_type, name='w')
        b = tf.constant([2.0, 3.0], dtype=float_type, name='b')
        mat1 =  integral_cos_mat(w=w, lplus=lplus, b=b).numpy()
        
        self.gp.random_weights_ = tf.transpose(w)
        self.gp.random_offset_ = b
        mat2 = self.gp._LowRankRFF__integral_mat(lplus = lplus).numpy()
        
        self.assertTrue((mat2 == mat1).all())
        self.assertAlmostEqual(mat1[0,0], 0.3012653529971747, places=7)
        self.assertAlmostEqual(mat1[1,1], 0.5542608460089069 , places=7)
        self.assertAlmostEqual(mat1[1,0], 0.3420353429585846 , places=7)


if __name__ == '__main__':
    unittest.main()








