# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np
import time

import tensorflow as tf

from point.model import InHomogeneousSpatialModel



class Results():
    
    def __init__(self):
            self.l2 = []
            self.loglik_fit = []
            self.loglik_test = []
            self.time = []
            
    def __str__(self):
        str_msg = "Results: ########## \n"
        str_msg += "average.lambda_mean.l2:= [%f] " % (sum(self.l2) / len(self.l2))
        str_msg +=  "\n"
        str_msg += "average.lambda_mean.loglikelihood.fitt:= [%f] " % (sum(self.loglik_fit) / len(self.loglik_fit))
        str_msg +=  "\n"
        str_msg += "average.lambda_mean.loglikelihood.test:= [%f] " % (sum(self.loglik_test) / len(self.loglik_test))
        str_msg +=  "\n"
        str_msg += "average.fitting_time:= [%f] " % (sum(self.time) / len(self.time))
        return str_msg
            
            

class Evaluation():

    def __init__(self, model, generator):
        
        #if not isinstance(generator, InHomogeneousSpatialModel):
            #raise ValueError("generator must be a 'InHomogeneousSpatialModel' instance")
            
        self.results = Results()
        self._model =  model
        self._generator = generator
        

    def run(self, optim_func, n_samples = 10, verbose = False):
        
        if not callable(optim_func):
            raise TypeError(
                "The 'closure' argument is expected to be a callable object."
            )  # pragma: no cover
            
        index_lst = np.arange(0, n_samples, 1).tolist()
        self._X = self._generator.generate(n_samples)
  
        for i in range(n_samples):
            
            print("start.sample@" + str(i+1))

            X_train = self._X[i]
            
            self._model.set_X(X_train)
            
            try:
                t0 = time.time()
                optim_func(self._model, verbose = False)
                opt_time = time.time() - t0
            except :
                print('sample#%i : ERROR stopped' %(i+1)) 
                print("")
                continue
      
            loglik = self._model.log_likelihood(X_train)
            l2 = tf.reduce_sum((self._model.predict_lambda(self._generator._grid) - self._generator._lambda)**2, 0)[0]
            
            self.results.loglik_fit.append(loglik.numpy())
            self.results.l2.append(l2.numpy())
            self.results.time.append(opt_time)
            
            sum_test_loglik = 0
            for k in np.delete(index_lst, i):
                sum_test_loglik += self._model.log_likelihood(self._X[k])
                
            self.results.loglik_test.append(sum_test_loglik / len(index_lst))
            
            print("")
                








