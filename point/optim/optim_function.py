# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:11:33 2021

@author: jesel
"""

import time
import numpy as np
from point.laplace import opt_type




def get_optim_func_rff(n_loop = 2, maxiter = None, xtol = None, lmin = 1e-05, lmax = 10.0, smax = None, num_attempt = 5, direct_grad = False) :
    
    
    _opt = opt_type.AUTO_DIFF
    if direct_grad is True : _opt = opt_type.DIRECT

    def optim_func(model, X, verbose = False):

        beta = np.sqrt(X.shape[0] / model.lrgp.space_measure)
        model.lrgp.set_drift(beta, trainable = True)
        model.set_X(X)

        t0 = time.time()
        init_params = model.lrgp.copy_initial_parameters()

        ℓmin_active = True
        ℓmax_active = True
        s_active = True
        n_iter = 0
        
        while (ℓmin_active or ℓmax_active or s_active) and n_iter < num_attempt :

            if verbose and n_iter > 0 : 
                print("optim_restart_" + str(n_iter)) 
                
            model.lrgp.reset_initial_parameters(init_params, sample = True)
            
            #try :
            model.optimize_mode(optimizer = opt_type.DIRECT, tol = xtol, verbose = verbose)

            for i in range(n_loop):
                model.optimize(optimizer = _opt, maxiter = maxiter, verbose = verbose)
                model.optimize_mode(optimizer = opt_type.DIRECT, maxiter = maxiter, tol = xtol, verbose = verbose) 
            
            # except BaseException :
            #     print("ERROR_RESTART ")
            #     continue
 
            ℓmin_active = np.any(model.lrgp.lengthscales.numpy() < lmin )
            ℓmax_active = np.any(model.lrgp.lengthscales.numpy() > lmax )
            s_active = (smax is not None) and (model.smoothness_test().numpy() > smax)
            n_iter +=1
            
        t = time.time() - t0
        t = t /(n_iter)

        if verbose :
            print("SLBPP(" + model.name + ") finished_in := [%f] " % (t))

            
        return t

    return optim_func



def get_optim_func_nyst(set_variance = False, maxiter = None, xtol = None, set_beta = False, preset_indexes = None) :
    
    def optim_func(model, X, verbose = False):
    
        t0 = time.time()

        beta = np.sqrt(X.shape[0] / model.lrgp.space_measure)
        
        if set_variance : model.lrgp.variance.assign(beta**2)
        if set_beta : model.lrgp.set_drift(beta, trainable = False)
        
        model.set_X(X)
        model.lrgp.set_data(model._X)
        
        if model.lrgp.n_components> X.shape[0] :  
            model.lrgp.set_sampling_data_with_replacement()
        elif preset_indexes is not None :
             model.lrgp._preset_data_split(preset_indexes)
        
        model.lrgp.fit()
        model.optimize_mode(maxiter = maxiter, tol = xtol, verbose = verbose) 
        t = time.time() - t0
        
        if verbose : 
            print("LBPP(" + model.name + ") finished_in := [%f] " % (t))
    
        return t
    
    return optim_func