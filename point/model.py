
from point.utils import check_random_state_instance
from scipy.interpolate import griddata

import numpy as np

class PointsData():
    
    def __init__(self, sizes, points, space):
        self.space = space
        self.points = points
        self.sizes = sizes
    
    @property
    def n_samples(self):
        return self.points.shape[0]
    
    @property
    def size(self, index  = 0):
        return len(self.points[index])
    
    def __getitem__(self, item):
        return self.points[item][0:self.sizes[item],:]




class  CoxLowRankSpatialModel() :
    
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

    
    def log_likelihood(self, X): 
        return self.lrgp.maximum_log_likelihood_objective(X)
    
    
    
    
    
class HomogeneousSpatialModel() :

    def __init__(self, lam, bound, random_state = None):
        self._lambda = lam
        self.bound = bound
        self._random_state = random_state
        
    def generate(self):
        random_state = check_random_state_instance(self._random_state)
        lam = self._lambda *  (self.bound[1] - self.bound[0])**2
       
        n_points = random_state.poisson(size= 1, lam = lam)
        points = random_state.uniform( self.bound[0], self.bound[1], size=(n_points[0], 2))
        return points
    
    
class InHomogeneousSpatialModel() :
    
    def __init__(self, grid, lam, bound, random_state = None):
        self._grid = grid
        self._lambda = lam
        self._random_state = random_state
        self.bound = bound
   
    def generate(self, n_samples = 1):
        random_state = check_random_state_instance(self._random_state)
        lambdaMax = max(self._lambda)
        
        hsm = HomogeneousSpatialModel(lambdaMax, self.bound, random_state= random_state)
        
        points_lst = []
        sizes = []
        n_max = 0
        
        for i in range(n_samples):
            full_points  = hsm.generate()
            lambdas =  griddata(self._grid, self._lambda.numpy(), full_points, method='nearest')
            
            n_lambdas = lambdas.shape[0]
            u = random_state.uniform(0, 1, n_lambdas)
            tmp = (u < lambdas.reshape(n_lambdas)/lambdaMax)
            
            retained_points = full_points[tmp]
            n_points = retained_points.shape[0]
            
            n_max = max(n_max , n_points)  
            points_lst.append(full_points[tmp])
            sizes.append(n_points)
            
        if n_samples == 1 :
            return retained_points
        
        #padding for the output
        points = np.zeros((n_samples, n_max, 2))
        for b in range(n_samples):
            points[b, :points_lst[b].shape[0]] = points_lst[b]

        return PointsData(
            sizes = sizes, 
            points = points, 
            space = self.bound
        )
    
    


    
    

    









