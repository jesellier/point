
import tensorflow as tf
from point.misc import Space
import gpflow

from gpflow.base import Parameter
from gpflow.utilities import positive

import abc
import copy


class LowRankBase(gpflow.models.GPModel, metaclass=abc.ABCMeta):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 250, n_features = 2,  random_state = None):

        super().__init__(
            kernel,
            likelihood=None,  # custom likelihood
            num_latent_gps = 1
        )
        
        
        self.n_components = n_components
        self.n_features = n_features 
        self.space = space
        
        if beta0 is None :
            self.beta0 = Parameter(1e-10, transform=positive(), name = "beta0")
            self.hasDrift = False
            gpflow.set_trainable(self.beta0, False)
        else :
            self.beta0 = Parameter(beta0, transform=positive(), name = "beta0")
            self.hasDrift = True
        
        self._is_fitted = False
        self._random_state = random_state
        
        
    def set_drift(self, value, trainable = True):
        gpflow.set_trainable(self.beta0, trainable)
        self.beta0.assign(value)
        self.hasDrift = True
        
    def copy_initial_parameters(self):
        tpl = copy.deepcopy((self.variance, self.lengthscales, self.beta0))
        return tpl
    
    def reset_initial_parameters(self, p, sample = True):
        self.variance.assign(p[0])
        self.lengthscales.assign(p[1])
        self.beta0.assign(p[2])
        if sample : self.fit(sample = True)
        
    @property
    def space_measure(self):
        return self.space.measure(self.n_features)
        
    @property
    def lengthscales(self):
        return self.kernel.lengthscales
    
    @property
    def variance(self):
        return self.kernel.variance
   
    @property
    def gradient_adjuster(self, include_drift = True):
        adjv = ( (tf.exp(self.variance) - 1)/ tf.exp(self.variance)) 
        adjl = ((tf.exp( self.lengthscales) - 1) / tf.exp( self.lengthscales))
        adjuster = tf.experimental.numpy.vstack([tf.expand_dims(adjl,1), tf.expand_dims(adjv,0)])
        
        if self.beta0.trainable is True and include_drift is True :
            adjb = ( (tf.exp(self.beta0) - 1)/ tf.exp(self.beta0)) 
            adjuster = tf.experimental.numpy.vstack([tf.expand_dims(adjb,0), adjuster])

        return adjuster
    

    def validate_entry(self, X):
        if len(X.shape) == 1:
            n_features = X.shape[0]
            X = tf.reshape(X, (1, n_features))
        else :
            _, n_features = X.shape
        return X
         
 
    def lambda_func(self, X):
        return (self.func(X) + self.beta0)**2


    def func(self, X) :
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        X = self.validate_entry(X)
        return self.feature(X) @ self._latent
    
    
    def maximum_log_likelihood_objective(self, X):
        integral_term = self.integral()
        data_term = sum(tf.math.log(self.lambda_func(X)))
        out = - integral_term + data_term 
        return out
    
    
    def predict_f(self, Xnew):
        raise NotImplementedError

    @abc.abstractmethod 
    def fit(self, sample = True):
        raise NotImplementedError()
        
    @abc.abstractmethod 
    def sample(self, latent_only = False):
        raise NotImplementedError()
        
    @abc.abstractmethod 
    def feature(self, X):
        raise NotImplementedError()
 
    @abc.abstractmethod 
    def __call__(self, X):
         raise NotImplementedError()

    @abc.abstractmethod 
    def integral(self, bound = None):
         raise NotImplementedError()
    
    @abc.abstractmethod 
    def M(self, bound = None):
        raise NotImplementedError()
        
    @abc.abstractmethod 
    def m(self, bound = None):
        raise NotImplementedError()



    
