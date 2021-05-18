import numpy as np
import arrow
import sys
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from gpflow.config import default_float
import gpflow.kernels as gfk 

from scipy.optimize import minimize #For optimizing

from point.utils import check_random_state_instance
from point.low_rank.low_rank_rff import LowRankRFF
from point.misc import Space, TensorMisc



class PointsData():
    
    def __init__(self, sizes, point_locations, space, trainable_variables = None, grad = None):
        self.space = space
        
        self.locs = point_locations
        self.sizes = sizes
      
        self.grad = grad
        self.variables = trainable_variables
        
    @property
    def num_sequences(self):
        return self.locs.shape[0]
        
    def shuffled_index(self, n_batch, random_state = None):
        random_state = check_random_state_instance(random_state)
        shuffled_idx = np.arange(self.num_sequences)
        random_state.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[- n_batch :]
        return shuffled_idx
        
    def plot_points(self, batch_index = 0):
        plt.figure()
        size = self.sizes[batch_index]
        plt.scatter(self.locs[batch_index][0:size,0], self.locs[batch_index][0:size,1], edgecolor='b', facecolor='none', alpha=0.5 );
        plt.xlim(-1, 1); plt.ylim(-1, 1)
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.show()
        
    def __getitem__(self, item):
        return self.locs[item][0:self.sizes[item],:]
            

class HomogeneousSpatialModel() :

    def __init__(self, lam, space, random_state = None):
        self._lambda = lam
        self.space = space
        
        self._random_state = random_state
        
    def generate(self):
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bounds
        lam = self._lambda *  self.space.measure
       
        n_points = random_state.poisson(size= 1, lam = lam)
        points = random_state.uniform( bounds[:, 0], bounds[:, 1], size=(n_points[0], bounds.shape[0]))
        return points
    


    
class CoxLowRankSpatialModel() :
    
    def __init__(self, low_rank_gp, random_state = None):
        self.lrgp =  low_rank_gp
        self.space = low_rank_gp.space
        
        self._random_state = random_state


    @property
    def parameters(self):
        return self.lrgp.parameters
        
    @property
    def trainable_variables(self):
        return self.lrgp.trainable_variables


    def sample(self):
        self.lrgp.sample()


    def compute_loss_and_gradients(self, points):
        if not self.lrgp._is_fitted :
            raise ValueError("lowRankGP instance not fitted")

        with tf.GradientTape() as tape:  
            self.lrgp.fit(sample = False)
            loss = self.lrgp.maximum_log_likelihood_objective(points)

        grad = tape.gradient(loss, self.trainable_variables) 
    
        return (loss, grad)
        


    def __optimizeBound(self, n_warm_up = 10000, n_iter = 0):
        
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bounds

        max_fun = 0
        res_max = None
        
        def objective_function(x):
             out = self.lrgp.lambda_func(tf.convert_to_tensor([[x[0], x[1]]], dtype=default_float()))
             out = out[0][0]
             return out.numpy()
        
        # Warm up with random points
        if n_warm_up is not None and n_warm_up > 0 :
            x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warm_up, bounds.shape[0]))
            fs = self.lrgp.lambda_func(tf.convert_to_tensor(x_tries, dtype=default_float())) 
            fs = fs.numpy()
            max_fun = fs.max()
        
        # Explore the parameter space more throughly
        if n_iter > 0:
            x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0]))
            x_seeds = np.append(x_seeds, [self.space.center], axis=0)
            res_max = None
        
            for x_try in x_seeds:
                # Find the minimum of minus the acquisition function
                res = minimize(lambda x: - objective_function(x), x_try, bounds=bounds)
        
                # See if success
                if not res.success:
                    continue
        
                # Store it if better than previous minimum(maximum).
                if max_fun is None or - res.fun >= max_fun:
                    max_fun = - res.fun
                    res_max = res

        return (max_fun, res_max)
    
    
        
    def generate(self, batch_size =1, n_warm_up = 10000, n_iter = 0, do_clipping = True, calc_grad = False, verbose = False):

        random_state = check_random_state_instance(self._random_state)

        points_lst = []
        grad_lst = []
        sizes = []
    
        max_len = 0
        n_gen = 0
        
        while n_gen < batch_size :
            self.sample()
            
            lambdaMax = self.__optimizeBound(n_warm_up = n_warm_up, n_iter = n_iter)[0]
            
            if lambdaMax > 1000 :
                print("OOM : generation skipped, cause Lambda:= " + str(lambdaMax))
                continue
 
            full_points  = HomogeneousSpatialModel(lambdaMax, self.space, random_state= random_state).generate()
      
            lambdas =  self.lrgp.func(tf.constant(full_points, dtype=default_float()))**2
            lambdas = lambdas.numpy()
            
            n_lambdas = lambdas.shape[0]
            if do_clipping is True : lambdas = np.clip(lambdas, 0, lambdaMax)

            u = random_state.uniform(0, 1, n_lambdas)
            tmp = (u < lambdas.reshape(n_lambdas)/lambdaMax)
            retained_points = full_points[tmp]
    
            n_points = retained_points.shape[0]
            max_len = max(max_len , n_points)  

            points_lst.append(retained_points)
            sizes.append(n_points)
            
            if calc_grad  :
               loss, grad = self.compute_loss_and_gradients(retained_points)
               grad = TensorMisc().pack_tensors(grad)
               grad_lst.append(grad)

            if verbose :
                print("[%s] %d-th sequence generated: %d raw samples. %d samples have been retained. " % \
                      (arrow.now(), n_gen+1, n_lambdas, n_points), file=sys.stderr)
                    
            n_gen += 1
                    
        # padding for the output
        points = np.zeros((batch_size, max_len, 2))
        for b in range(batch_size):
            points[b, :points_lst[b].shape[0]] = points_lst[b]
        
        return PointsData(
            sizes = sizes, 
            point_locations = points, 
            space = self.space.bounds1D,  
            trainable_variables = np.array(self.parameters, dtype=object), 
            grad = tf.stack(grad_lst)
            )

    

if __name__ == "__main__":
    
    rng = np.random.RandomState()
    sp = Space([-1,1])
    
    variance = tf.Variable([8], dtype=default_float(), name='sig')
    length_scale = tf.Variable([0.5], dtype=default_float(), name='l')
    kernel = gfk.SquaredExponential(variance= variance , lengthscales= length_scale)
    beta0 = tf.Variable([0.5], dtype=default_float(), name='beta0')

    lrgp = LowRankRFF(kernel, beta0 = beta0, space = sp, n_components =  250, random_state = rng)
    lrgp.fit()

    X = tf.constant(rng.normal(size = [10, 2]), dtype=default_float(), name='X')
    process = CoxLowRankSpatialModel(lrgp, random_state = rng)
    data = process.generate(verbose = False, n_warm_up = 10000, batch_size =10, calc_grad = True)

    process.lrgp.plot_kernel()
    process.lrgp.plot_surface()
    data.plot_points()
    


        
    
    