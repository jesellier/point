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
import time

from gpflow.config import default_float

from point.helper import get_lrgp, method
from point.model import Space

rng = np.random.RandomState(40)



def expandedSum(x, n =0):
    z1 = tf.expand_dims(x, 1)
    z2 = tf.expand_dims(x, 0)

    return (z1 + z2, z1 - z2)


def termB(z, bounds = [-1,1]):
    
    z = tf.transpose(z)
    z1 = z[:, 0]
    z2 = z[:, 1]

    bound = bounds[1]
    Mp, _ = expandedSum(z)
    d1 = Mp[:,:,0] 
    d2 = Mp[:,:,1]

    M = (2 / (d1 * d2)) * tf.sin(bound*d1)  * tf.sin(bound*d2) 
    diag = (1 /  (2 * z1* z2)) * tf.sin(2*bound*z1) * tf.sin(2 *bound*z2)                               
    M = tf.linalg.set_diag(M, diag) 

    return M



def termB_der(z, l, bounds = [-1,1]):

    z1 = z[0, :]
    z2 = z[1, :]

    bound = bounds[1]
    
    l1 = l[0]
    l2 = l[1]

    Mp, _ = expandedSum(tf.transpose(z))
    d1 = Mp[:,:,0] 
    d2 = Mp[:,:,1]

    M = (2 / (d1 * d2)) * tf.sin(bound*d1)  * tf.sin(bound*d2) 
    diag = (1 /  (2 * z1* z2)) * tf.sin(2*bound*z1) * tf.sin(2 *bound*z2)                                
    M = tf.linalg.set_diag(M, diag) 

    diagB = diag
    dl1 = - ( 2 * bound * tf.cos(bound*d1) * tf.sin(bound*d2) / d2 - M ) / l1
    dl1 =  tf.linalg.set_diag(dl1, - (  bound * tf.cos(2*bound*z1) * tf.sin(2*bound*z2) / z2 - diagB) / l1)  
      
    dl2 = - ( 2 * bound * tf.sin(bound*d1) * tf.cos(bound*d2) / d1 - M ) / l2
    dl2 =  tf.linalg.set_diag(dl2, - ( bound * tf.sin(2*bound*z1) * tf.cos(2*bound*z2) / z1 - diagB) / l2) 

    out = tf.experimental.numpy.vstack([tf.expand_dims(M,0), tf.expand_dims(dl1,0), tf.expand_dims(dl2,0)])
    
    return out



class Test_B(unittest.TestCase):
    
    
    def grad_recalc(self, z, cache1, cache2):
        l = self.l
        with tf.GradientTape() as tape:  
                gamma = 1 / (2 * l **2 )
                z = tf.linalg.diag(tf.math.sqrt(2 * gamma)) @ self.G
                M = termB(z)
                loss = tf.transpose(cache1) @ M @ cache2
        grad = tape.gradient(loss, l) 
        return (loss, grad)
    
    
    def setUp(self):
        self.v = tf.Variable(1, dtype=default_float(), name='sig')
        self.l = tf.Variable([0.2, 0.5], dtype=default_float(), name='l')
        self.gamma = 1 / (2 * self.l **2 )
        self.G = tf.constant([[1, 0.5],[1, 0.5]], dtype=default_float(), name='w')



    def test_values(self):
        bounds = [-1, 1.0]
        z = tf.linalg.diag(tf.math.sqrt(2 * self.gamma)) @ self.G
        # i.e. z = d(2) x size_features i.e.  two features [5, 2] and [2.5, 1]

        M = termB(z, bounds).numpy()

        self.assertAlmostEqual( (1 / 20) * np.sin(10) * np.sin(4), M[0,0], places=7)
        self.assertAlmostEqual( 0.020585826711063307, M[0,0], places=7)

        self.assertAlmostEqual( (1 / 5) * np.sin(5) * np.sin(2), M[1,1], places=7)
        self.assertAlmostEqual(  -0.17438947509437502, M[1,1], places=7)

        self.assertAlmostEqual( (4/45) * np.sin(7.5) * np.sin(3), M[0,1], places=7)
        self.assertAlmostEqual( 0.011766272380676126, M[0,1], places=7)

        
    
    def test_grad(self):
        
        z = tf.linalg.diag(tf.math.sqrt(2 * self.gamma)) @ self.G
        # i.e. z = d(2) x size_features i.e.  two features [5, 2] and [2.5, 1]

        M, grad1, grad2 = termB_der(z, self.l)
        
        #grad M[0,0]
        cache1 = tf.expand_dims(tf.constant([1, 0], dtype=default_float()),1)
        cache2 = tf.expand_dims(tf.constant([1, 0], dtype=default_float()),1)
        loss, grad_tf = self.grad_recalc(z, cache1, cache2)
        self.assertAlmostEqual( loss.numpy(), M[0,0], places=7)
        self.assertAlmostEqual( grad1[0,0].numpy(), grad_tf[0].numpy(), places=7)
        self.assertAlmostEqual( grad1[0,0].numpy(), -1.48459943 , places=7)
        
        self.assertAlmostEqual( grad2[0,0].numpy(), grad_tf[1].numpy(), places=7)
        self.assertAlmostEqual( grad2[0,0].numpy(), -0.10106672 , places=7)
        
        #grad M[1,1]
        cache1 = tf.expand_dims(tf.constant([0, 1], dtype=default_float()),1)
        cache2 = tf.expand_dims(tf.constant([0, 1], dtype=default_float()),1)
        loss, grad_tf = self.grad_recalc(z, cache1, cache2)
        self.assertAlmostEqual( loss.numpy(), M[1,1], places=7)
        self.assertAlmostEqual( grad1[1,1].numpy(), grad_tf[0].numpy(), places=7)
        self.assertAlmostEqual( grad1[1,1].numpy(), -2.16161385 , places=7)
        
        self.assertAlmostEqual( grad2[1,1].numpy(), grad_tf[1].numpy(), places=7)
        self.assertAlmostEqual( grad2[1,1].numpy(), -0.66802159 , places=7)
        
        #grad M[1,1]
        cache1 = tf.expand_dims(tf.constant([1, 0], dtype=default_float()),1)
        cache2 = tf.expand_dims(tf.constant([0, 1], dtype=default_float()),1)
        loss, grad_tf = self.grad_recalc(z, cache1, cache2)
        self.assertAlmostEqual( loss.numpy(), M[1,0], places=7)
        self.assertAlmostEqual( grad1[1,0].numpy(), grad_tf[0].numpy(), places=7)
        self.assertAlmostEqual( grad1[1,0].numpy(), -0.1042259 , places=7)
        
        self.assertAlmostEqual( grad2[1,0].numpy(), grad_tf[1].numpy(), places=7)
        self.assertAlmostEqual( grad2[1,0].numpy(), 0.51879278 , places=7)
        
        #grad M[1,1]
        cache1 = tf.expand_dims(tf.constant([0, 1], dtype=default_float()),1)
        cache2 = tf.expand_dims(tf.constant([1, 0], dtype=default_float()),1)
        loss, grad_tf = self.grad_recalc(z, cache1, cache2)
        self.assertAlmostEqual( loss.numpy(), M[0,1], places=7)
        self.assertAlmostEqual( grad1[0,1].numpy(), grad_tf[0].numpy(), places=7)
        self.assertAlmostEqual( grad1[0,1].numpy(), -0.1042259 , places=7)
        
        
        self.assertAlmostEqual( grad2[0,1].numpy(), grad_tf[1].numpy(), places=7)
        self.assertAlmostEqual( grad2[0,1].numpy(), 0.51879278 , places=7)
        
        
        


class Test_Quadratic_der(unittest.TestCase):
    
    
    def setUp(self):
        self.v = tf.Variable(1, dtype=default_float(), name='sig')
        self.l = tf.Variable([0.2, 0.5], dtype=default_float(), name='l')
        self.gamma = 1 / (2 * self.l **2 )
        self.G = tf.constant([[1, 0.5],[1, 0.5]], dtype=default_float(), name='w')
        self.verbose = True

    @unittest.SkipTest
    def test(self):

       lrgp = get_lrgp(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-1,1]), n_components = 2, random_state = rng)
       lrgp._G = self.G
       lrgp.fit(sample = False)
       
       cache = tf.ones((2 * lrgp.n_components,1),  dtype=default_float())
       lrgp._latent = cache
       
       #### TF GRADIENT
       t0 = time.time()
       with tf.GradientTape() as tape:  
           lrgp.fit(sample = False)
           intM = tf.transpose(lrgp._latent) @ lrgp.M() @ lrgp._latent
       grad_tf = tape.gradient(intM, lrgp.trainable_variables) 

       adjv = (tf.exp(self.v)/ (tf.exp(self.v) - 1))
       adjl = tf.expand_dims((tf.exp( self.l)/(tf.exp( self.l) - 1)),1)
       
       grad_tf_l = tf.expand_dims(grad_tf[0],1)*adjl
       grad_tf_l = grad_tf_l[:,0]
       grad_tf_v = grad_tf[1]*adjv
 
       if self.verbose :
           print("TF loss:= %f - in [%f] " % (intM,time.time() - t0))
           print(grad_tf_v)
           print(grad_tf_l)
           print("")
           
       #### IMPLEMENTED GRADIENT
       t0 = time.time()
       (intD, grad) = lrgp.integral(get_grad = True)
       
       grad_v = grad[2]
       grad_l = (grad[0:2])[:,0]
       
       if self.verbose :
           print("Implementation loss:= %f - in [%f] " % (intD, time.time() - t0))
           print(grad_v)
           print(grad_l)

       intM = intM[0][0].numpy()
       
       #### TEST
       #test loss values
       self.assertAlmostEqual(intM, intD.numpy(), places=7)
       self.assertAlmostEqual(intM, 4.805755111166799, places=7)
       
       #test gradient variance
       self.assertAlmostEqual(grad_v.numpy(),grad_tf_v.numpy() , places=7)
       self.assertAlmostEqual(grad_v[0].numpy(), 4.80575511, places=7)
       
       #test gradient l
       self.assertAlmostEqual(grad_v.numpy(),grad_tf_v.numpy() , places=7)
       self.assertAlmostEqual(grad_l[0].numpy(), 17.5115577, places=7)
       self.assertAlmostEqual(grad_l[1].numpy(), 0.57677141, places=7)
       
       
class Test_features_der(unittest.TestCase):
        
        def setUp(self):
            self.v = tf.Variable(1, dtype=default_float(), name='sig')
            self.l = tf.Variable([0.2, 0.5], dtype=default_float(), name='l')
            self.gamma = 1 / (2 * self.l **2 )
            self.G = tf.constant([[1, 0.5],[1, 0.5]], dtype=default_float(), name='w')
            
            X = np.array( [[-1.37923991,  1.37140879],
                           [ 0.02771165, -0.32039958],
                           [-0.84617041, -0.43342892],
                           [-1.3370345 ,  0.20917217],
                           [-1.4243213 , -0.55347685],
                           [ 0.07479864, -0.50561983],
                           [ 1.05240778,  0.97140041],
                           [ 0.07683154, -0.43500078],
                           [ 0.5529944 ,  0.26671631],
                           [ 0.00898941,  0.64110275]])
            
            self.X = tf.convert_to_tensor(X, dtype=default_float())
            self.verbose = True
            
        #@unittest.SkipTest
        def test(self):
            
            lrgp = get_lrgp(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-1,1]), n_components = 2, random_state = rng)
            lrgp._G = self.G
            lrgp.fit(sample = False)

            #TF : compute the quadratic term ones x features x ones
            N = self.X.shape[0]
            cache1 = tf.expand_dims(tf.experimental.numpy.hstack([tf.ones(N,  dtype=default_float())]) ,0)
            cache2 = tf.expand_dims(tf.experimental.numpy.hstack([tf.ones(2 * lrgp.n_components,  dtype=default_float())]) ,1)
            
            t0 = time.time()
            with tf.GradientTape() as tape:  
                lrgp.fit(sample = False)
                loss_tf =  cache1 @ lrgp.feature(self.X) @ cache2
            grad_tf = tape.gradient(loss_tf, lrgp.trainable_variables) 
     
            if self.verbose is True :
               print("TF loss:= %f - in [%f] " % (loss_tf, time.time() - t0))
               print( grad_tf[0])
               print( grad_tf[1])
       
            #Recalculation 
            grad_adj = lrgp.gradient_adjuster
            out, grads = lrgp.feature(self.X, get_grad = True)
            print(out)
    
            loss = cache1 @ out @ cache2
            dl1 =  grad_adj[0] * (cache1 @ grads[0] @ cache2)
            dl2 =  grad_adj[1] * (cache1 @ grads[1]  @ cache2)
            dv = grad_adj[2] * (cache1 @ grads[2] @ cache2)
 
            if self.verbose :
                print("")
                print("Implementation loss:= %f - in [%f] " % (loss, time.time() - t0))
                print(tf.experimental.numpy.hstack([dl1, dl2]))
                print(dv)
    
            #### TEST
            self.assertAlmostEqual(loss_tf[0], loss[0], places=7)
            self.assertAlmostEqual(loss[0][0].numpy(), 2.4085244420859335  , places=7)
 
            self.assertAlmostEqual(grad_tf[1].numpy(), dv[0][0].numpy(), places=7)
            self.assertAlmostEqual(dv[0][0].numpy(),  0.7612389081418001 , places=7)

            self.assertAlmostEqual(dl1[0][0].numpy(), grad_tf[0][0].numpy(), places=7)
            self.assertAlmostEqual(dl1[0][0].numpy(), -2.8832875507335376 , places=7)
            
            self.assertAlmostEqual(dl2[0][0].numpy(), grad_tf[0][1].numpy(), places=7)
            self.assertAlmostEqual(dl2[0][0].numpy(), 5.388804478660982, places=7)
           
           
   
       

       
 

       
       
    

        


        
        



        

   



if __name__ == '__main__':
    unittest.main()








