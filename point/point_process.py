import numpy as np
import arrow
import sys
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

from scipy.optimize import minimize #For optimizing

from point.utils import check_random_state_instance
from point.low_rank_rff import LowRankRFF
from point.low_rank_nystrom import LowRankNystrom
from point.misc import Space

import gpflow.kernels as gfk


class PointsData():
    
    def __init__(self, sizes, point_locations, space, trainable_variables = None, loglik = None, grad = None):
        self.batch_size = len(sizes)
        self.space = space
        
        self.sizes = sizes
        self.locs = point_locations

        self.grad = grad
        self.loglik = loglik
        self.variables = trainable_variables
        
        
class TransformerLogExp():
    def __init__(self):
        pass
    
    def __call__(self, theta):
        return tf.math.log(tf.math.exp(theta) - 1)
    
    def inverse(self, theta):
        return tf.math.log(tf.math.exp(theta) + 1)
     
        

class HomogeneousSpatialModel() :

    def __init__(self, lam, random_state = None):
        self.lambda_ = lam
        self.random_state = random_state
        
    def generate(self, sp = Space()):
        random_state = check_random_state_instance(self.random_state)
        n_points = random_state.poisson(size= 1, lam =self.lambda_ * sp.measure)
        points = random_state.uniform(sp.bounds[:, 0], sp.bounds[:, 1], size=(n_points[0], sp.bounds.shape[0]))
        return points
    


    
class CoxLowRankSpatialModel() :
    
    def __init__(self, low_rank_gp, random_state = None):
        self.lrgp =  low_rank_gp
        self.random_state = random_state


    @property
    def parameters(self):
        return self.lrgp.parameters
        
    @property
    def trainable_variables(self):
        return self.lrgp.trainable_variables

    
    def sample(self):
        self.lrgp.sample()
    

    def likelihood_grad(self, points, lplus = 1, lminus = 0):
        if not self.lrgp.is_fitted :
            raise ValueError("lowRankGP instance not fitted")

        with tf.GradientTape() as tape:  
            self.lrgp.fit(sample = False)
            out = self.lrgp.likelihood(points, lplus = lplus, lminus = lminus)

        grad = tape.gradient(out, self.trainable_variables) 
        
        return (out, grad)
        


    def __optimizeBound(self, n_warm_up = 10000, n_iter = 0, sp = Space()):
        
        random_state = check_random_state_instance(self.random_state)
        bounds = sp.bounds

        max_fun = 0
        res_max = None
        
        def __func(x):
             out = self.lrgp.func(tf.constant([x[0], x[1]], dtype=float_type))
             out = out[0][0]
             return out.numpy()
        
        # Warm up with random points
        if n_warm_up is not None and n_warm_up > 0 :
            x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warm_up, bounds.shape[0]))
            
            self.x_tries = x_tries
            fs = self.lrgp.func(tf.constant(x_tries, dtype=float_type))**2
            fs = fs.numpy()
            
            max_fun = fs.max()

            #x_max = x_tries[fs.argmax()]
        
        # Explore the parameter space more throughly
        if n_iter > 0:
            x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0]))
            x_seeds = np.append(x_seeds, [sp.center], axis=0)
            res_max = None
        
            for x_try in x_seeds:
                # Find the minimum of minus the acquisition function
                res = minimize(lambda x: - (__func(x)**2), x_try, bounds=bounds)
        
                # See if success
                if not res.success:
                    continue
        
                # Store it if better than previous minimum(maximum).
                if max_fun is None or - res.fun >= max_fun:
                    max_fun = - res.fun
                    res_max = res

        return (max_fun, res_max)
    
    
        
    def generate(self, sp = Space(), batch_size =1, n_warm_up = 10000, n_iter = 0, do_clipping = True, calc_grad = False, verbose = False):

        random_state = check_random_state_instance(self.random_state)

        points_list = []
        sizes = []
        grad_list = []
        loglik = []
        
        max_len = 0
        
        for b in range(batch_size) :
            self.sample()
            lambdaMax = self.__optimizeBound(n_warm_up = n_warm_up, n_iter = n_iter, sp = sp)[0]
            
            if lambdaMax > 1000 :
                raise ValueError("OOM Lambda:= " + str(lambdaMax))
 
            full_points  = HomogeneousSpatialModel(lambdaMax * sp.measure, random_state= random_state).generate(sp)
      
            lambdas =  self.lrgp.func(tf.constant(full_points, dtype=float_type))**2
            lambdas = lambdas.numpy()
            
            n_lambdas = lambdas.shape[0]
            if do_clipping is True : lambdas = np.clip(lambdas, 0, lambdaMax)
            
            u = random_state.uniform(0, 1, n_lambdas)
            tmp = (u < lambdas.reshape(n_lambdas)/lambdaMax)
            retained_points = full_points[tmp]
            #thinned = X[~tmp]
            
            n_points = retained_points.shape[0]
            max_len = max(max_len , n_points)  

            points_list.append(retained_points)
            sizes.append(n_points)
            
            if calc_grad :
               out, grad = self.likelihood_grad(
                   retained_points, 
                   lplus = sp._higher_bounds, 
                   lminus = sp._lower_bounds
                   )
               
               loglik .append(out)
               grad_list.append(grad)

            if verbose :
                print("[%s] %d-th sequence generated: %d raw samples. %d samples have been retained. " % \
                      (arrow.now(), b+1, n_lambdas, n_points), file=sys.stderr)
                    
        # padding for the output
        points = np.zeros((batch_size, max_len, 2))
        for b in range(batch_size):
            points[b, :points_list[b].shape[0]] = points_list[b]
        
        return PointsData(sizes, points, [sp._lower_bounds, sp._higher_bounds],
                          np.array(self.trainable_variables, dtype=object), 
                          loglik, 
                          np.array(grad_list, dtype=object)
                          )
    
    

def print_points(x1):
    plt.scatter(x1[:,0], x1[:,1], edgecolor='b', facecolor='none', alpha=0.5 );
    plt.xlim(-1, 1); plt.ylim(-1, 1)
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()
    
    
    
if __name__ == "__main__":
    rng = np.random.RandomState()

    variance = tf.Variable(5, dtype=float_type, name='sig')
    length_scale = tf.Variable([0.5,0.5], dtype=float_type, name='l')
    
    X = tf.constant(rng.normal(size = [500, 2]), dtype=float_type, name='X')
    
    space = Space(-1,1)
    bounds = space.bounds
    
    lrgp = LowRankRFF(length_scale , variance, n_components = 250, random_state = rng).fit()

    #kernel = gfk.SquaredExponential(variance= variance ** 2 , lengthscales= length_scale)
    #lrgp = LowRankNystrom(kernel, n_components = 250, random_state = rng, noise = 1e-5, mode = 'sampling').fit()
    
    p = CoxLowRankSpatialModel(lrgp, random_state = rng)
    data = p.generate(verbose = False, sp = space) #, calc_grad = True)

    out, grad = p.likelihood_grad(X)
    
    print(out)
    print(grad)
    print(data.sizes)

    print_points(data.locs[0])
    p.lrgp.plot_surface()


    


        
    
    