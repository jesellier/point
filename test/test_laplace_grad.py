# -*- coding: utf-8 -*-

import numpy as np
import time
import unittest

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from gpflow.config import default_float

from point.misc import Space, TensorMisc
from point.helper import get_process, method
from point.model import Space
from point.laplace import LaplaceApproximation

rng = np.random.RandomState(40)




    
def _log_marginal_likelihood(lp):

    lp.set_latent(lp._mode)
    Qmode = - tf.linalg.inv(lp.get_w_hessian())
    #Qmode +=  1e-5* tf.eye(lp.n_components, dtype=default_float()) # add jitter

    grad_int, grad_second, grad_det = _get_explicit_der(lp, Qmode)
    grad_int *= lp.lrgp.gradient_adjuster
    grad_second *= lp.lrgp.gradient_adjuster
    grad_det *= lp.lrgp.gradient_adjuster
    grad = grad_int + grad_second + grad_det
        
    return (grad, grad_int + grad_second, grad_det)

 

def _get_explicit_der(lp, Qmode):
    "explicit grad of marginal_likelihood w.r.t theta"

    #integral term
    out, grad = lp.lrgp.integral(get_grad = True)
    out = - out  - 0.5 * tf.norm(lp.latent )**2
    grad_int = - grad
    
    # second loglik term
    features, grad_feat = lp.lrgp.feature(lp.X, get_grad = True)
    sum_grad =  grad_feat @ lp.latent
    f = features @ lp.latent
    grad_second = tf.expand_dims(2 * tf.reduce_sum(tf.transpose(sum_grad[:,:,0]) / f, 0),1)
    out += sum(tf.math.log(lp.lrgp.lambda_func(lp.X)))[0]
    
    #log det
    out += - 0.5 * tf.linalg.logdet(Qmode)

    #term1
    _, grad_m = lp.lrgp.M(get_grad = True)
    grad_det = tf.expand_dims(tf.reduce_sum(grad_m * Qmode, [1,2]),1)  #equivalent to tf.linalg.trace(grad @ Q)

    #term3
    a = features @ tf.transpose(Qmode) 
    trace_term = tf.expand_dims(tf.reduce_sum(a *  features,1),1) #equivalent to tf.linalg.trace( [features @ features.T] @ Q)
    grad_det += - 2 * tf.transpose(tf.transpose(sum_grad[:,:,0]) / (f)**3 ) @ trace_term 

    #term2
    trace_d1 = tf.reduce_sum(a * grad_feat, 2) #equivalent to tf.linalg.trace([grad_f @ festures.T] @ Q)
    trace_d2 = tf.reduce_sum((grad_feat @ tf.transpose(Qmode) ) * features, 2) #equivalent to tf.linalg.trace([features @ grad_f.T] @ Q)
    grad_det += (trace_d1 + trace_d2) @ (1 / f**2)
    
    return ( grad_int, grad_second, grad_det)





class Test_Posterior_Grad(unittest.TestCase):

    def setUp(self):
        self.v = tf.Variable(1, dtype=default_float(), name='sig')
        self.l = tf.Variable([0.5, 0.5], dtype=default_float(), name='l')
        
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


    @unittest.SkipTest
    def test_grad_posterior(self):
        
       model = get_process(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-10,10]), n_components = 10, random_state = rng)
       model.lrgp.fit(sample = False)
       lp = LaplaceApproximation(model, self.X)    

       #TF
       lp.set_latent(tf.Variable(lp.latent))
       t0 = time.time()
       with tf.GradientTape() as tape:  
            loss_tf = lp.log_posterior()
       grad_tf = tape.gradient(loss_tf, lp.latent) 
       
       if self.verbose :
           print("TEST grad_posterior")
           print(grad_tf[:,0])
           print("in " + str(time.time() - t0))
           print("")
       
       #Implementation
       grads = lp.get_w_grad()
       
       if self.verbose :
           print("IMPLEMENTED grad_posterior")
           print(grads[:,0])
           print("in " + str(time.time() - t0))
           print("")
       
       #TEST
       for i in range(lp.n_components):
           self.assertAlmostEqual(grads[i,0].numpy(), grad_tf[i,0].numpy(), places=7)
           

        
           
class Test_MLL_Grad(unittest.TestCase):

    def setUp(self):
        self.v = tf.Variable(1, dtype=default_float(), name='sig')
        self.l = tf.Variable([0.5, 0.5], dtype=default_float(), name='l')
        
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
        
        mode = np.array([[-0.47575364],
            [ 0.20223473],
            [-0.07979932],
            [-0.54016023],
            [ 0.46790774],
            [-0.18703021],
            [ 1.63257138],
            [-1.071316  ],
            [-0.72459877],
            [-0.94876385],
            [ 0.34305699],
            [ 0.25685773],
            [-0.93180307],
            [ 0.73835176],
            [-1.44086932],
            [-0.90916212],
            [-0.3605749 ],
            [ 0.00653856],
            [-0.44359125],
            [ 1.28144293]])
        
        
        G = np.array([[-0.6075477, -0.12613641, -0.68460636, 0.92871475, -1.84440103, -0.46700242, 2.29249034,  0.48881005,  0.71026699,  1.05553444],
             [ 0.0540731, 0.25795342, 0.58828165 , 0.88524424, -1.01700702, -0.13369303, -0.4381855, 0.49344349, -0.19900912, -1.27498361]])
        
        
        self.X = tf.convert_to_tensor(X, dtype=default_float())
        self.G = tf.convert_to_tensor(G, dtype=default_float())
        self.mode = tf.convert_to_tensor(mode)
        self.verbose = True


    #@unittest.SkipTest
    def test_grad_total_mll(self):
        
       model = get_process(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-10,10]), n_components = 10, random_state = rng)
       model.lrgp._G = self.G
       model.lrgp.fit(sample = False)
       
       lp = LaplaceApproximation(model, self.X) 
       lp.set_mode(self.mode)

       #TF
       t0 = time.time()
       with tf.GradientTape() as tape:  
           model.lrgp.fit(sample = False)
           loss_tf  = lp.log_marginal_likelihood()
       grad_tf = tape.gradient(loss_tf, lp.lrgp.trainable_variables) 
       grad_tf = tf.expand_dims(TensorMisc().pack_tensors(grad_tf),1)

       if self.verbose :
           print("TEST grad_mll")
           print(grad_tf)
           print("in " + str(time.time() - t0))
           print("")
       
       #Implementation
       t0 = time.time()
       loss, grads = lp.log_marginal_likelihood(get_grad = True)
       
       if self.verbose :
           print("IMPLEMENTED grad_mll")
           print(grads)
           print("in " + str(time.time() - t0))
           print("")

       #TEST
       for i in range(3):
           self.assertAlmostEqual(grads[i,0].numpy(), grad_tf[i,0].numpy(), places=5)
           
           
    #@unittest.SkipTest
    def test_loglik_grad(self):
        
       model = get_process(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-10,10]), n_components = 10, random_state = rng)
       model.lrgp.fit(sample = False)
       model.lrgp._G = self.G
       
       lp = LaplaceApproximation(model, self.X) 
       lp.set_mode(self.mode)
       gradient_adjuster = lp.lrgp.gradient_adjuster


       #GRAD integral compare
       t0 = time.time() 
       with tf.GradientTape() as tape:  
           model.lrgp.fit(sample = False)
           loss = - lp.model.lrgp.integral()
       grad_tf1= tape.gradient(loss, model.lrgp.trainable_variables) 
       grad_tf1 = tf.expand_dims(TensorMisc().pack_tensors(grad_tf1),1)
    
       if self.verbose :
           print("TF integral.term.grad")
           print(grad_tf1)
           print("in " + str(time.time() - t0))
           print("")
 
       t0 = time.time() 
       integral, grad1 = lp.model.lrgp.integral(get_grad = True)
       integral = - integral
       grad1 = - grad1
       grad1 *= gradient_adjuster 
       
       if self.verbose :
           print("IMPLEMENTED integral.term.grad")
           print(grad1)
           print("in " + str(time.time() - t0))
           print("")
           

       #SECOND TERM compare
       t0 = time.time() 
       with tf.GradientTape() as tape:  
            lp.lrgp.fit(sample = False)
            loss2 =  sum(tf.math.log(lp.lrgp.lambda_func(self.X)))
       grad_tf2 = tape.gradient(loss2, lp.lrgp.trainable_variables) 
       grad_tf2 = tf.expand_dims(TensorMisc().pack_tensors(grad_tf2),1)
        
       if self.verbose :
           print("TF loglik.second.term.grad")
           print(grad_tf2)
           print("in " + str(time.time() - t0))
           print("")
    
       t0 = time.time() 
       features, grad_feat = lp.lrgp.feature(self.X, get_grad = True)
       sum_grad =  grad_feat @  lp.latent
       f = features @ lp.latent
       grad2 = tf.expand_dims(2 * tf.reduce_sum(tf.transpose(sum_grad[:,:,0]) / f, 0),1)
       grad2 *= gradient_adjuster 
       
       second_term = sum(tf.math.log(lp.lrgp.lambda_func(self.X)))[0]
            
       if self.verbose :
           print("IMPLEMENTED loglik.second.term.grad")
           print(grad2)
           print("in " + str(time.time() - t0))
           print("")

       #TEST
       self.assertAlmostEqual(integral.numpy(), loss.numpy(), places=7)
       self.assertAlmostEqual(second_term.numpy(), loss2.numpy(), places=7)
       
       for i in range(3):
           self.assertAlmostEqual(grad1[i,0].numpy(), grad_tf1[i,0].numpy(), places=5)
           self.assertAlmostEqual(grad2[i,0].numpy(), grad_tf2[i,0].numpy(), places=5)
           
           
           
    #@unittest.SkipTest
    def test_log_det(self):
        
       model = get_process(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-10,10]), n_components = 10, random_state = rng)
       model.lrgp._G = self.G
       model.lrgp.fit(sample = False)

       lp = LaplaceApproximation(model, self.X) 
       lp.set_mode(self.mode)
       
       gradient_adjuster = lp.lrgp.gradient_adjuster
       Qmode  = lp._Qmode 
       
       
       # LOG.DET with TF
       t0 = time.time()
       with tf.GradientTape() as tape:  
            model.lrgp.fit(sample = False)
            H = lp.get_w_hessian()
            Q  = - tf.linalg.inv(H)
            loss =   -0.5 * tf.linalg.logdet(Q)
       grad_tf1 = tape.gradient(loss, lp.lrgp.trainable_variables) 
       grad_tf1 = tf.expand_dims(TensorMisc().pack_tensors(grad_tf1),1)
       
       if self.verbose :
           print("TF brute.force log.det.grad")
           print(grad_tf1)
           print("in " + str(time.time() - t0))
           print("")

       t0 = time.time()
       A = tf.constant(Qmode)
       with tf.GradientTape() as tape:  
            model.lrgp.fit(sample = False)
            H = lp.get_w_hessian()
            loss = - 0.5 *  tf.expand_dims(tf.reduce_sum(H * A, 1),1)#tf.linalg.trace(Qinv @ A)
       grad_tf2 = tape.gradient(loss, lp.lrgp.trainable_variables) 
       grad_tf2 = tf.expand_dims(TensorMisc().pack_tensors(grad_tf2),1)
       
       if self.verbose :
           print("TF log.det.grad")
           print(grad_tf2)
           print("in " + str(time.time() - t0))
           print("")
       
       # LOG.DET Implemented
       t0 = time.time()
       features, grad_feat = lp.lrgp.feature(self.X, get_grad = True)
       sum_grad =  grad_feat @  lp.latent
       f = features @ lp.latent
       
       #term1
       _, grad_m = lp.lrgp.M(get_grad = True)
       grad = tf.expand_dims(tf.reduce_sum(grad_m * Qmode, [1,2]),1)  #equivalent to tf.linalg.trace(grad @ Q)

       #term3
       a = features @ tf.transpose(Qmode) 
       trace_term =  tf.expand_dims(tf.reduce_sum(a *  features,1),1) #equivalent to tf.linalg.trace( [features @ features.T] @ Q)
       grad += - 2 * tf.transpose(tf.transpose(sum_grad[:,:,0]) / (f)**3 ) @ trace_term 

       #term2
       trace_d1 = tf.reduce_sum(a * grad_feat, 2) #equivalent to tf.linalg.trace([grad_f @ festures.T] @ Q)
       trace_d2 = tf.reduce_sum((grad_feat @ tf.transpose(Qmode) ) * features, 2) #equivalent to tf.linalg.trace([features @ grad_f.T] @ Q)
       grad += (trace_d1 + trace_d2) @ (1 / f**2)
       grad *= gradient_adjuster
       
       if self.verbose :
           print("Implemented log.det.grad")
           print(grad)
           print("in " + str(time.time() - t0))
           print("")
 
       #TEST
       for i in range(3):
          self.assertAlmostEqual(grad_tf1[i,0].numpy(), grad_tf2[i,0].numpy(), places=5)
          self.assertAlmostEqual(grad_tf1[i,0].numpy(), grad[i,0].numpy(), places=5)


if __name__ == '__main__':
    unittest.main()
    




    
    
# #%%             GRAD LOG DET Q wrt W

# H =  lp.get_w_hessian()
# Qinv = - H
# Q  = - tf.linalg.inv(H)

# def marginal_t():
#     H = lp.get_w_hessian()
#     Q  = - tf.linalg.inv(H)
#     return  -0.5 * tf.linalg.logdet(Q)

# x_max = tf.Variable(lp.arg_mode)
# lp.set_latent(x_max)

# t0 = time.time()
# with tf.GradientTape() as tape:  
#     model.lrgp.fit(sample = False)
#     loss = marginal_t()
# grad1 = tape.gradient(loss, x_max) 
# print("finished in " + str(time.time() - t0))

# t0 = time.time()
# A = tf.constant(Q)
# with tf.GradientTape() as tape:  
#     model.lrgp.fit(sample = False)
#     H = lp.get_w_hessian()
#     loss2 = - 0.5 *  tf.expand_dims(tf.reduce_sum(H * A, 1),1)#tf.linalg.trace(Qinv @ A)
# grad2 = tape.gradient(loss2, x_max) 
# print("finished in " + str(time.time() - t0))

# t0 = time.time()
# with tf.GradientTape() as tape:  
#     model.lrgp._latent = x_max
#     model.lrgp.fit(sample = False)
#     weights = 1 / (features  @ lrgp._latent)**2
#     out = tf.expand_dims(tf.linalg.trace(tf.expand_dims(features,2) @ tf.expand_dims(features,1) @ A),1)
#     loss3 = tf.transpose(weights) @ out
# grad3 = tape.gradient(loss3, x_max) 
# print("finished in " + str(time.time() - t0))


# # TEST
# t0 = time.time()
# features = lrgp.feature(X)
# weights = features / (features  @ lrgp._latent)**3 

# #Brute Force
# #trace_term = tf.expand_dims(tf.linalg.trace(tf.expand_dims(features,2) @ tf.expand_dims(features,1) @ Q),1)
# #Better
# #featM = tf.expand_dims(features,2) @ tf.expand_dims(features,1)
# #trace_term = tf.expand_dims(tf.reduce_sum(featM * Q, [1,2]),1)
# #even better
# a = features @ tf.transpose(Q) 
# trace_term =  tf.expand_dims(tf.reduce_sum(a *  features,1),1)
# print("finished in " + str(time.time() - t0))
# grad4 =- 2 * tf.transpose(weights) @ trace_term


# #%%             GRAD LOG DET Q wrt theta

# H =  lp.get_w_hessian()
# Qinv = - H
# Q  = - tf.linalg.inv(H)

# def marginal_t():
#     H = lp.get_w_hessian()
#     Q  = - tf.linalg.inv(H)
#     return  -0.5 * tf.linalg.logdet(Q)

# x_max = tf.constant(lp.arg_mode)
# lp.set_latent(x_max)

# t0 = time.time()
# A = tf.constant(Q)
# with tf.GradientTape() as tape:  
#     model.lrgp.fit(sample = False)
#     loss =  marginal_t()
# grad = tape.gradient(loss, lrgp.trainable_variables) 
# #print("finished in " + str(time.time() - t0))
# #print(grad)

# t0 = time.time()
# A = tf.constant(Q)
# with tf.GradientTape() as tape:  
#     model.lrgp.fit(sample = False)
#     H = lp.get_w_hessian()
#     loss1 = - 0.5 *  tf.expand_dims(tf.reduce_sum(H * A, 1),1)#tf.linalg.trace(Qinv @ A)
# grad1 = tape.gradient(loss1, lrgp.trainable_variables) 
# #print("finished in " + str(time.time() - t0))
# #print(grad1)
# #print("")

# t0 = time.time()
# A = tf.constant(Q)
# with tf.GradientTape() as tape:  
#     model.lrgp.fit(sample = False)
#     M = lrgp.M()
#     m = tf.linalg.trace(M @ A)
# grad2 = tape.gradient(m, lrgp.trainable_variables) 
# #print("finished in " + str(time.time() - t0))
# #print(grad2)
# #print("")

# t0 = time.time()
# A = tf.constant(Q)
# with tf.GradientTape() as tape:  
#     model.lrgp.fit(sample = False)
#     features = lrgp.feature(X)
#     V = features / (features  @ x_max )
#     term =  tf.transpose(V) @ V
#     loss3 = tf.expand_dims(tf.reduce_sum( term * A, 1),1)#tf.linalg.trace(Qinv @ A)
# grad3 = tape.gradient(loss3, lrgp.trainable_variables) 
# #print("finished in " + str(time.time() - t0))
# #print(grad1)
# #print("")


# t0 = time.time()
# term1
# f = features  @ lrgp._latent
# _, grad = lrgp.M(get_grad = True)
# #term1 = tf.expand_dims(tf.linalg.trace(grad @ A),1)
# term1 = tf.expand_dims(tf.reduce_sum(grad * A, [1,2]),1)
# #term1 = term1 * adj
# print(term1)

# #term3
# features, grad_feat = lrgp.feature(X, get_grad = True)
# weights =  grad_feat @ lrgp._latent
# weights = tf.transpose(weights[:,:,0]) / (f)**3 
# term3 =- 2 * tf.transpose(weights) @ trace_term 
# #term3 = term3 * adj
# print(term3)

# #term2
# a = features @ tf.transpose(Q) 
# trace2 = tf.reduce_sum(a * grad_feat, 2)

# b = grad_feat @ tf.transpose(Q) 
# trace3 = tf.reduce_sum(b * features, 2)
# t = trace2 + trace3

# weights = 1 / (f)**2
# term2 = t @ weights
# print(term2)
# #term2 = term2 * adj
# #print(term2 + term3)

# grad = term1 + term2 + term3
# #print("finished in " + str(time.time() - t0))
# #print(grad)


# #%%
# #LAST TEST : bench
# t0 = time.time()
# x_max  = tf.constant( x_max )
# lrgp._latent = x_max

# M =  2 * model.lrgp.M() +  tf.eye(150,  dtype= default_float())
# features = model.lrgp.feature(X)
# f = features  @ x_max
# a =  tf.expand_dims( tf.reduce_sum(features / f, 0), 1)
# grad = - M @ x_max + 2 * a
# print(grad)
# print("")
# print(0.5* M @ x_max)
# print("")
# print(a)
# print("")

# # TOTAL
# with tf.GradientTape() as tape:  
#     lrgp.fit(sample = False)
#     M1 =  lrgp.M() + 0.5 * tf.eye(150,  dtype= default_float())
#     M1inv  = tf.linalg.inv(M1)
#     features = lrgp.feature(X)
#     f = features  @ x_max 
#     A =  tf.expand_dims( tf.reduce_sum(features / f, 0), 1)
#     loss1 = 2 * M1inv @ A
#     loss1 = sum(loss1)
# grad1 = tape.gradient(loss1, lrgp.trainable_variables) 
# print(grad1)
# print("")

# # TERM1
# u = tf.constant( Minv @ a )
# B = tf.constant( Minv  )
# with tf.GradientTape() as tape:  
#     lrgp.fit(sample = False)
#     M1 =  lrgp.M() + 0.5 * tf.eye(150,  dtype= default_float())
#     loss3 = - 2 * B @ M1 @ u
#     loss3 = sum(loss3)
# grad3 = tape.gradient(loss3, lrgp.trainable_variables) 
# print(grad3)
# print("")

# #TERM2
# u = tf.constant( Minv @ a )
# B = tf.constant( Minv  )
# with tf.GradientTape() as tape:  
#     lrgp.fit(sample = False)
#     features = lrgp.feature(X)
#     f = features  @ x_max 
#     A =  tf.expand_dims( tf.reduce_sum(features / f, 0), 1)
#     loss2 = 2 * B @ A
#     loss2 = sum(loss2)
# grad2 = tape.gradient(loss2, lrgp.trainable_variables) 
# print("TERM_2")
# print(grad2)
# print("")

# #SUBTERM1
# u = tf.constant( Minv @ a )
# B = tf.constant( Minv )

# f = features  @ x_max 
# num = tf.constant(f )
# with tf.GradientTape() as tape:  
#     lrgp.fit(sample = False)
#     features = lrgp.feature(X)
#     A =  tf.expand_dims( tf.reduce_sum(features / f, 0), 1)
#     loss2 = 2 * B @ A
#     loss2 = sum(loss2)
# grad2 = tape.gradient(loss2, lrgp.trainable_variables) 
# print("SUBTERM_21")
# print(grad2)
# print("")

# #SUBTERM2
# u = tf.constant( Minv @ a )
# B = tf.constant( Minv )

# f = features  @ x_max 
# dem = tf.constant(features )
# with tf.GradientTape() as tape:  
#     lrgp.fit(sample = False)
#     features = lrgp.feature(X)
#     f = features  @ x_max 
#     A =  tf.expand_dims( tf.reduce_sum(dem / f, 0), 1)
#     loss2 = 2 * B @ A
#     loss2 = sum(loss2)
# grad2 = tape.gradient(loss2, lrgp.trainable_variables) 
# print("SUBTERM_22")
# print(grad2)
# print("")




# ################# term 1
# M, grad_m =  lrgp.M(get_grad = True)
# grad_m = 2 * grad_m
# #term1 = - Minv @ grad_m @ Minv @ a
# #term1 = tf.reduce_sum(term1[:,:,0], 1)
# #term1 = tf.expand_dims(term1,1) * adj
# #print(term1)
# #print("")

# term1 = - Minv @ grad_m @ x_max
# term1 = tf.reduce_sum(term1[:,:,0], 1)
# term = tf.expand_dims(term1,1) * adj
# #print(term12)
# #print("")

# ################# term 2
# features, grad_f = lrgp.feature(X, get_grad = True)
# term21 =  tf.reduce_sum(grad_f / f, 1)
# out21 =  2 * Minv @ tf.transpose(term21)
# out21 =  tf.expand_dims(tf.reduce_sum(tf.transpose(out21), 1),1) * adj

# weights =  grad_feat @ lrgp._latent
# weights = tf.transpose(weights[:,:,0]) / f**2
# weights= tf.expand_dims(tf.transpose(weights),2)
# term22 = tf.reduce_sum(weights * features, 1)
# out22 = - 2 * Minv @ tf.transpose(term22)
# out22 = tf.expand_dims(tf.reduce_sum(tf.transpose(out22), 1),1) * adj

# v = term22 + term21
# vec = (grad_m @ x_max)[:,:,0] - 2 * v
# mat =  Q @ M +   0.5 * tf.eye(lp.n_components,  dtype= default_float())
# mat =  M @ tf.linalg.inv(mat) @ Q
# mat = Q - Q @ mat
# dw = tf.transpose(v @ mat)
# print(dw)



# #%%
# n = 100
# m = 50
# a = np.random.rand(m, n)
# b = np.random.rand(n, m)

# t0 = time.time()
# M = tf.expand_dims(features,2) @ tf.expand_dims(features,1)
# trace_term = tf.expand_dims(tf.reduce_sum(M * Q, [1,2]),0)
# print("finished in " + str(time.time() - t0))
# print(trace_term)

# t0 = time.time()
# trace_term = tf.expand_dims(tf.linalg.trace(tf.expand_dims(features,2) @ tf.expand_dims(features,1) @ Q),1)
# print("finished in " + str(time.time() - t0))
# print(trace_term)

# # They all should give the same result
# print(np.trace(a.dot(b)))
# print(np.sum(a*b.T))