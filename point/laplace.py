# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from gpflow.config import default_float

from point.optim.minimize_line_search import OptimLineSearch
from point.optim.minimize_scipy_autodiff import OptimScipyAutoDiff
from point.optim.minimize_scipy import OptimScipy



class  LaplaceApproximation() :
    
    def __init__(self, model, X = None):
        self.model =  model
        self.space = model.lrgp.space
        self._random_state = self.model.lrgp._random_state
        self.X = X
        self.jitter = 1e-5
        self._is_fitted = False
        
    @property
    def n_components(self):
        return self.lrgp._latent.shape[0]

    @property
    def latent(self):
        return self.model.lrgp._latent
    
    @property
    def lrgp(self):
        return self.model.lrgp
        
    
    def set_latent(self, latent):
        self.model.lrgp._latent = latent
        pass
    
    def set_X(self, X):
        self._X = X
        pass
    
    def set_mode(self, mode):
        self.set_latent(mode)
        self._mode = mode
        self._Qmode  = - tf.linalg.inv(self.get_w_hessian())
        pass
    
    
    def posterior_latent_distribution(self):
               
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        return tfp.distributions.MultivariateNormalTriL(loc= self._mode[:,0], scale_tril=tf.linalg.cholesky(self._Qmode))
    
    
    def predict_f(self, Xnew):
        
        if not self._is_fitted :
            raise ValueError("instance not fitted")

        features = self.lrgp.feature(Xnew)
        mean =  features  @ self.latent + self.lrgp.beta0
        var = tf.expand_dims(tf.linalg.diag_part(features @ self._Qmode @ tf.transpose(features)),1)
        return mean, var
    
    
    def predict_lambda(self, Xnew):
        # f ~ Normal(mean_f, var_f)
        mean_f, var_f = self.predict_f(Xnew)
        # λ = E[f²] = E[f]² + Var[f]
        lambda_mean = mean_f ** 2 + var_f

        return lambda_mean
    
    
    def log_likelihood(self, Xnew):
        return self.model.log_likelihood(Xnew)[0]


    def log_posterior(self):
        return self.model.log_likelihood(self._X) - 0.5 * tf.norm(self.latent )**2
    
    
    def log_marginal_likelihood(self, get_grad = False, adjust_der = True):
        
        if not self._is_fitted :
            raise ValueError("instance not fitted")

        Qmode = - tf.linalg.inv(self.get_w_hessian())
        self._Qmode  = Qmode

        if get_grad is True :
            out, grad = self.__get_explicit_der()
            #grad += self.__get_implicit_der()
            
            if adjust_der is True : 
                grad *= self.lrgp.gradient_adjuster
            
            return (out, grad)

        return - 0.5 * tf.linalg.logdet(Qmode) + self.log_posterior() 


    def __get_explicit_der(self):
        "explicit grad of marginal_likelihood w.r.t theta"

        #integral term 
        #     out := - latent^T @ M @ latent - 0.5* latent^T@latent
        #     grad:= - latent^T @ der_M @ latent
        out, grad = self.lrgp.integral(get_grad = True)
        out = - out  - 0.5 * tf.norm(self.latent )**2
        grad = - grad
        
        #loglikelihood data term
        #     out:= sum [log (|features @ latent|^2) ]
        #     grad:= 2 * sum [(der_features @ latent) / (features @ latent)]
        features, grad_feat = self.lrgp.feature(self._X, get_grad = True)
        sum_grad =  grad_feat @ self.latent
        f = features @ self.latent
        grad += tf.expand_dims(2 * tf.reduce_sum(tf.transpose(sum_grad[:,:,0]) / f, 0),1)
        out += sum(tf.math.log(self.lrgp.lambda_func(self._X)))[0]
        
        #log det term
        out += - 0.5 * tf.linalg.logdet(self._Qmode)

        #compute the der of the log det term
        #term1 : grad = tr [der_M @ Q ] 
        _, grad_m = self.lrgp.M(get_grad = True)
        grad += tf.expand_dims(tf.reduce_sum(grad_m * self._Qmode, [1,2]),1)  #equivalent to tf.linalg.trace(grad_M @ Q)

        #term2 : grad = sum[ ((der_features @ latent) / (features @ latent)^2  * tr[ features @ Q @ features^T] ] ]
        a = features @ tf.transpose(self._Qmode) 
        trace_term =  tf.expand_dims(tf.reduce_sum(a *  features,1),1) #equivalent to tf.linalg.trace( [features @ features.T] @ Q)
        grad += - 2 * tf.transpose(tf.transpose(sum_grad[:,:,0]) / (f)**3 ) @ trace_term 

        #term3 : grad = sum[ (1/(features @ latent)^2) * tr[ der[features @ features^T] @ Q ] ]
        trace_d1 = tf.reduce_sum(a * grad_feat, 2) #equivalent to tf.linalg.trace([grad_f @ festures.T] @ Q)
        trace_d2 = tf.reduce_sum((grad_feat @ tf.transpose(self._Qmode) ) * features, 2) #equivalent to tf.linalg.trace([features @ grad_f.T] @ Q)
        grad += (trace_d1 + trace_d2) @ (1 / f**2)
        
        return (out, grad)
    

    def __get_implicit_der(self):
        "explicit grad of marginal_likelihood w.r.t theta"

        #redondant terms
        features, grad_feat = self.lrgp.feature(self.X, get_grad = True)
        a = features @ tf.transpose(self._Qmode) 
        trace_term =  tf.expand_dims(tf.reduce_sum(a *  features,1),1) 
        f = features @ self.latent
        M, grad_m = self.lrgp.M(get_grad = True)
        
        #d.det/dw part
        grad_det = - 2 * tf.transpose(features / (features  @ self.latent)**3 ) @ trace_term
        
        #dw/dtheta part
        v =  tf.reduce_sum(grad_feat / f, 1)
        weights = grad_feat @ self.latent
        weights = tf.transpose(weights[:,:,0]) / f**2
        weights= tf.expand_dims(tf.transpose(weights),2)
        v += tf.reduce_sum(weights * features, 1)

        v = (grad_m @ self.latent)[:,:,0] - 2 * v
        mat =  self._Qmode @ M +   0.5 * tf.eye(self.n_components,  dtype= default_float())
        mat =  M @ tf.linalg.inv(mat) @ self._Qmode
        mat = self._Qmode - self._Qmode @ mat
        dw = tf.transpose(v @ mat)
        
        #total
        grad = tf.reduce_sum( tf.transpose(dw * grad_det), 1, keepdims= True)

        return grad


    def get_w_grad(self):
        "grad of posterior w.r.t latent"
        M =  self.lrgp.M() + 0.5 * tf.eye(self.n_components,  dtype= default_float())
        features = self.lrgp.feature(self._X)
        grad = - 2 * M @ self.lrgp._latent + 2 * tf.expand_dims( tf.reduce_sum(features /  (features  @ self.lrgp._latent), 0), 1)
        return grad
    
    
    def get_w_hessian(self):
        "hessian of posterior w.r.t latent"
        M =  self.lrgp.M() + 0.5 * tf.eye(self.n_components,  dtype= default_float())
        features = self.lrgp.feature(self._X)
        V = features / (features  @ self.lrgp._latent )
        H = -2 * ( M + tf.transpose(V) @ V)
        return H

   
    def optimize(self, optimizer = "scipy", restarts= 1, n_seeds= 10, maxiter = 100, verbose=True):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
        
        def objective_closure_der():
            self.lrgp.fit(sample = False)
            out, grad = self.log_marginal_likelihood(get_grad = True)
            return -1 * out, -1 * grad
        
        def objective_closure():
            self.lrgp.fit(sample = False)
            return -1 * self.log_marginal_likelihood()

        if optimizer == "line_search" :

            res = OptimLineSearch().minimize(
                objective_closure_der, 
                variables = self.lrgp.trainable_variables, 
                evaluation_function = objective_closure, 
                restarts= restarts, 
                n_seeds= n_seeds, 
                maxiter = maxiter, 
                verbose= verbose, 
                random_state = self._random_state)

        elif  optimizer == "scipy_autodiff" :
            res = OptimScipyAutoDiff().minimize(objective_closure, self.lrgp.trainable_variables, options = {'maxiter': maxiter})

        elif optimizer == "scipy" :
            res = OptimScipy().minimize(objective_closure_der, self.lrgp.trainable_variables, options = {'maxiter': maxiter})  
        
        
        self.lrgp.fit(sample = False)
        
        return res
   

    def _optimize_mode(self, optimizer = "scipy", restarts= 10, n_seeds= 10, maxiter = 100, verbose=True):
    
        if self._X is None :
            raise ValueError("data: X must be set")
  
        self.set_latent(tf.Variable(self.latent))
        
        def objective_closure():
             return -1 * self.log_posterior()
            
        def objective_closure_der():
            return -1 * self.log_posterior(), -1 * self.get_w_grad()

        if optimizer == "line_search" :
            
            OptimLineSearch().minimize(
                objective_closure_der, 
                variables = self.latent, 
                evaluation_function = objective_closure, 
                restarts= restarts, 
                n_seeds= n_seeds, 
                maxiter = maxiter, 
                verbose= verbose, 
                random_state = self._random_state)

        elif  optimizer == "scipy_autodiff" :
           OptimScipyAutoDiff().minimize(objective_closure, self.latent, options = {'maxiter': maxiter})

        elif optimizer == "scipy" :
            OptimScipy().minimize(objective_closure_der, self.latent, options = {'maxiter': maxiter})

        self.set_latent(tf.constant(self.latent))
        self._mode = tf.constant(self.latent)
        self._Qmode  = - tf.linalg.inv(self.get_w_hessian())
        self._is_fitted = True


