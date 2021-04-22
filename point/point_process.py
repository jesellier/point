import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from scipy.optimize import minimize #For optimizing

from point.utils import check_random_state_instance
from point.low_rank_gp import LowRankApproxGP

import time



class Space():
    def __init__(self, bounds = np.array(((0, 1), (0, 1))) ):
        super().__init__()
        self._bounds = bounds
        
        
    @property
    def x1Min(self):
        return self._bounds[0][0]
    
    @property
    def x1Max(self):
        return self._bounds[0][1]
    @property
    def x2Min(self):
        return self._bounds[1][0]
    @property
    def x2Max(self):
        return self._bounds[1][1]
        

    def measure(self):
        return (self.x2Max - self.x1Min) * (self.x2Max - self.x1Min)
        
    def center(self):
        return [(self.x1Min + self.x1Max)/2,(self.x2Min + self.x2Max)/2]
    
    def x1Bound(self):
        return self._bounds[0]
    
    def x2Bound(self):
        return self._bounds[1]
        
    

class HomogeneousSPP() :

    def __init__(self, lam, random_state = None):
        super().__init__()
        self.lambda_ = lam
        self.random_state = random_state
        
    def generate(self, sp = Space()):
        random_state = check_random_state_instance(self.random_state)
        numPoints = random_state.poisson(size= 1, lam =self.lambda_ * sp.measure())[0]
        X = random_state.uniform(sp._bounds[:, 0], sp._bounds[:, 1], size=(numPoints, sp._bounds.shape[0]))
        return X
    
    
    
    
class InhomogeneousSPP() :
    
    def __init__(self, functor, random_state = None):
        super().__init__()
        self.fun_lambda = functor
        self.lamMax = None
        self.random_state = random_state
        
    def optimizeBound(self, sp):
        resultsOpt=minimize(lambda x : - self.fun_lambda(x[0],x[1]), sp.center(), bounds=(sp.x1Bound(), sp.x2Bound()));
        self.lambdaMax = - resultsOpt.fun; #retrieve minimum value found by minimize
        
        
    def generate(self, sp = Space()):
        
        random_state = check_random_state_instance(self.random_state)
        self.optimizeBound(sp)
        
        X  = HomogeneousSPP(self.lambdaMax * sp.measure(),  random_state = random_state ).generate(sp)
        num = len(X)
        
        tmp = random_state .uniform(0,1, num)< self.fun_lambda(X[:,0], X[:,1])/self.lambdaMax 
        x_thinned = X[~tmp]
        x_points = X[tmp]

        return x_points, x_thinned
    
    
    
class CoxLowRankSGP() :
    
    def __init__(self, length_scale, variance = 1.0, n_components = 10, random_state = None):
        super().__init__()
        self.lrgp_ =  LowRankApproxGP(n_components, random_state)
        self.random_state = random_state
        self.length_scale = length_scale
        self.variance  = variance

    def __func(self, x):
        out = self.lrgp_.func(tf.constant([x[0], x[1]], dtype=float_type))
        out = out[0][0]
        return out.numpy()


        
    def optimizeBound(self, n_warm_up = 10000, n_iter = 30, sp = Space()):
        
        random_state = check_random_state_instance(self.random_state)
        bounds = sp._bounds

        if not self.lrgp_.is_fitted :
            self.lrgp_.fit(self.length_scale, self.variance)

        max_fun = 0

        # Warm up with random points
        if n_warm_up is not None and n_warm_up > 0 :
            x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warm_up, bounds.shape[0]))
            fs = self.lrgp_.func(tf.constant(x_tries, dtype=float_type))**2
            fs = fs.numpy()
            
            max_fun = fs.max()
            #x_max = x_tries[fs.argmax()]
        
        # Explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0]))
        x_seeds = np.append(x_seeds, [sp.center()], axis=0)
        res_max = None
        
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: - (self.__func(x)**2), x_try, bounds=bounds)
    
            # See if success
            if not res.success:
                continue
    
            # Store it if better than previous minimum(maximum).
            if max_fun is None or - res.fun >= max_fun:
                max_fun = - res.fun
                res_max = res
                
        self.lambdaMax = max_fun
            

        return res_max
    
    
        
    def generate(self, sp = Space(), do_clipping = True):

        random_state = check_random_state_instance(self.random_state)
        self.optimizeBound(sp = sp) 

        lambdaMax = self.lambdaMax
        
        X  = HomogeneousSPP(lambdaMax * sp.measure(), random_state= random_state ).generate(sp)
        lambdas =  self.lrgp_.func(tf.constant(X, dtype=float_type))**2

        lambdas = lambdas.numpy()
        n = lambdas.shape[0]
        if do_clipping is True : lambdas = np.clip(lambdas, 0, lambdaMax)

        u = random_state.uniform(0,1, n)
        tmp = (u < lambdas.reshape(n)/lambdaMax)

        x_thinned = X[~tmp]
        x_points = X[tmp]
        
        return x_points, x_thinned
        
        
if __name__ == "__main__" :
    
    rng = np.random.RandomState(40)
    
    t0 = time.time()
    sp = Space()
    bounds = sp._bounds
    variance = tf.Variable(100.0, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')

    p = CoxLowRankSGP(length_scale=length_scale, variance = variance, random_state = rng)
    x_points, x_thinned = p.generate()
    #print(p.lrgp_.randombeta_)
    #print(p.lrgp_.randomFourier_.random_weights_)
    #print(p.lrgp_.randomFourier_.random_offset_)

    
    #print("FINAL:" + str(time.time()-t0))

    #r1 = p.optimizeBound()
    
    #x_tries = rng.uniform(bounds[:, 0], bounds[:, 1], size=(10000, bounds.shape[0]))
    #fs = p.lrgp_.func(tf.constant(x_tries, dtype=float_type))**2
    #fs = fs.numpy()
    #max_fun = fs.max()
    #x_max = x_tries[fs.argmax()]
    #print(p.lambdaMax)
    
    
    
    