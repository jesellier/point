import numpy as np

#import tensorflow as tf
#import tensorflow_probability as tfp
#tfd = tfp.distributions
#tfk = tfp.math.psd_kernels

#float_type = tf.dtypes.float64

from point.utils import check_random_state_instance

from bayesoptim.utils import UtilityFunction
from bayesoptim.optimizer import  ModifiedBayesianOptimizer

from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor



class Space():
    def __init__(self, lower_bounds = 0, higher_bounds = 1):
        self._bounds = np.array(((lower_bounds,  higher_bounds ), (lower_bounds,  higher_bounds ))) 
        self._lower_bounds=  lower_bounds
        self._higher_bounds = higher_bounds
        
    
    @property
    def bounds(self):
        return self._bounds
    
        
    @property
    def measure(self):
        return (self.__x2Max() - self.__x1Min()) * (self.__x2Max() - self.__x1Min())
    
    @property
    def center(self):
        return [(self.__x1Min() + self.__x1Max())/2, (self.__x2Min() + self.__x2Max())/2]
 
    def __x1Min(self):
        return self._bounds[0][0]
    
    def __x1Max(self):
        return self._bounds[0][1]

    def __x2Min(self):
        return self._bounds[1][0]

    def __x2Max(self):
        return self._bounds[1][1]  


    
    
class PointsData():
    
    def __init__(self, sizes, point_locations):
        self.batch_size = len(sizes)
        
        self._sizes = sizes
        self._locs = point_locations

        
 

class HomogeneousSpatialModel() :

    def __init__(self, lam, random_state = None):
        super().__init__()
        self.lambda_ = lam
        self.random_state = random_state
        
    def generate(self, sp = Space()):
        random_state = check_random_state_instance(self.random_state)
        n_points = random_state.poisson(size= 1, lam =self.lambda_ * sp.measure)
        points = random_state.uniform(sp.bounds[:, 0], sp.bounds[:, 1], size=(n_points[0], sp.bounds.shape[0]))
        return points

    
class CoxSpatialModel() :
    
    def __init__(self, length_scale, variance = 1.0, random_state = None):
        super().__init__()
        self.random_state = random_state
        self._length_scale = length_scale
        self._variance = variance
        
        kernel = kernel = variance * RBF(length_scale= length_scale)
        
        self._gp = GaussianProcessRegressor(
            kernel= kernel,
            alpha=  1e-6,
            normalize_y= False,
            optimizer = None,
            random_state= random_state
        )
        


    def __optimizeBound(self, n_warm_up = 20, n_iter = 20, sp = Space()):
        
        random_state = check_random_state_instance(self.random_state)
        bounds = sp.bounds

        minmax_flag = 1.0
        flip = -1.0
        
        optimizer = ModifiedBayesianOptimizer(bounds, random_state)
        optimizer.setGP(self._gp)
        utility = UtilityFunction(kind="ucb", kappa=2.0, xi=0.0).utility
        
        if n_warm_up > 0 :
            x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warm_up, bounds.shape[0]))
            t_tries = self._gp.sample_y(x_tries)
            for i in range(len(x_tries)):
                optimizer.register(x_tries[i], t_tries[i][0])
        
        #main run
        for _ in range(n_iter) :
            point = optimizer.suggest(utility, minmax_flag)
            #args = {'x': point[0], 'y': point[1]}
            target = optimizer.gp_sample_y(point)
            optimizer.register(point, target)
            minmax_flag = flip * minmax_flag
            
        print(optimizer.max)
        print(optimizer.min)
        
        absMax = max(np.abs(optimizer.max['target']), np.abs(optimizer.min['target']))**2

        return absMax
    
    
        
    def generate(self, sp = Space(), batch_size =1, do_clipping = True, calc_grad = False, verbose = False):

        random_state = check_random_state_instance(self.random_state)

        points_list = []
        sizes = []

        max_len = 0
        
        for b in range(batch_size) :
            
            absMax = self.__optimizeBound(sp = sp)
            print(absMax)
            
            lambdaMax = absMax**2
            full_points  = HomogeneousSpatialModel(lambdaMax * sp.measure, random_state= random_state).generate(sp)
            print(len(full_points))
            
            print("SAMPLING START")
            lambdas =  self._gp.sample_y(full_points)**2
            print("SAMPLING DONE")
            
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

        # padding for the output
        points = np.zeros((batch_size, max_len, 2))
        for b in range(batch_size):
            points[b, :points_list[b].shape[0]] = points_list[b]
        
        return PointsData(sizes, points)
    
    

def print_points(x1):
    plt.scatter(x1[:,0], x1[:,1], edgecolor='b', facecolor='none', alpha=0.5 );
    plt.xlabel("x"); plt.ylabel("y");
    plt.show()
    
    
    
if __name__ == "__main__":
    rng = np.random.RandomState(10)
    variance = 0.2
    length_scale = 0.0002
    p = CoxSpatialModel(length_scale=length_scale, variance = variance, random_state = rng)

    sp = Space(-1,1)
    #data = p.generate(verbose = True, sp = sp)
    #print_points(data._locs[0])
    
  
    
    
    


        
    
    