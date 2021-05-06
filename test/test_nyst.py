# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64
rng = np.random.RandomState(40)

import unittest
import scipy.integrate as integrate

from point.helper import method, get_process
from point.misc import Space


def eigvalsh_to_eps(spectrum, cond=None, rcond=None):
        """
        Determine which eigenvalues are "small" given the spectrum.
        This is for compatibility across various linear algebra functions
        that should agree about whether or not a Hermitian matrix is numerically
        singular and what is its numerical matrix rank. This is designed to be compatible with scipy.linalg.pinvh.
        Parameters
        ----------
        spectrum : 1d ndarray
            Array of eigenvalues of a Hermitian matrix.
        cond, rcond : float, optional
            Cutoff for small eigenvalues.
            Singular values smaller than rcond * largest_eigenvalue are
            considered zero.
            If None or -1, suitable machine precision is used.
        Returns
        -------
        eps : float
            Magnitude cutoff for numerical negligibility.
        """
        if rcond is not None:
            cond = rcond
        if cond in [None, -1]:
            t = spectrum.dtype.char.lower()
            factor = {'f': 1E3, 'd': 1E6}
            cond = factor[t] * np.finfo(t).eps
        eps = cond * np.max(abs(spectrum))
        return eps
    
    
def tf_calc_Psi_matrix_SqExp(Z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z,z') = ∫ K(z,x) K(x,z') dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).
    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.
    Does not broadcast over leading dimensions.
    """
    variance = tf.cast(variance, Z.dtype)
    lengthscales = tf.cast(lengthscales, Z.dtype)

    mult = tf.cast(0.5 * np.sqrt(np.pi), Z.dtype) * lengthscales
    inv_lengthscales = 1.0 / lengthscales

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]

    z1 = tf.expand_dims(Z, 1)
    z2 = tf.expand_dims(Z, 0)

    zm = (z1 + z2) / 2.0

    exp_arg = tf.reduce_sum(-tf.square(z1 - z2) / (4.0 * tf.square(lengthscales)), axis=2)

    erf_val = tf.math.erf((zm - Tmin) * inv_lengthscales) - tf.math.erf(
        (zm - Tmax) * inv_lengthscales
    )
    product = tf.reduce_prod(mult * erf_val, axis=2)
    out = tf.square(variance) * tf.exp(exp_arg + tf.math.log(product))
    return out
    


class Test_Nystrom(unittest.TestCase):
     
    #def setUp(self):
    def setUp(self):
        rng = np.random.RandomState(5)
        variance = tf.Variable(8, dtype=float_type, name='sig')
        length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        space = Space(-1,1)
        self.process = get_process(length_scale = length_scale, variance = variance, 
                                   space = space, method = method.NYST, n_components = 250, random_state = rng)
        
        self.X = tf.constant(rng.normal(size = [10, 2]), dtype=float_type, name='X')


    def test_singularity(self):
        process = self.process
        s = process.lrgp._lambda.numpy()

        eps = eigvalsh_to_eps(s, None, None)
        d = s[s > eps]
        
        # if np.min(s) < -eps:
        #     raise ValueError('the input matrix must be positive semidefinite')
        # if len(d) < len(s) :
        #     raise np.linalg.LinAlgError('singular matrix')
            
        #the input matrix must be positive semidefinite
        self.assertTrue(not (np.min(s) < -eps))
        
        #the input matrix must be singular
        self.assertTrue(not (len(d) < len(s)))
        

    def test_func_recalculation(self):
        X = self.X
        lrgp = self.process.lrgp
        
        f = lrgp.func(X)
        f = f.numpy()

        feature = lrgp.feature(X)
        w = tf.linalg.diag(tf.math.sqrt(lrgp._lambda)) @ lrgp.latent_
        f_recalc = feature @ w
        f_recalc = f_recalc.numpy()

        for i in range(f.shape[0]):
            self.assertAlmostEqual(f[i][0], f_recalc[i][0], places=7)
            
              
    def test_eigenfunction_integral(self):
        lrgp = self.process.lrgp
        w = tf.linalg.diag(tf.math.sqrt(lrgp._lambda)) @ lrgp.latent_
        bounds = lrgp.space.bounds1D
        
        #x = self.X[0]
        index_i = 0
        index_j = 0

        U = lrgp._U
        #test = tf.math.reduce_sum(lrgp._impl_kernel(x, lrgp._x) * U[:, index_i]) * tf.math.reduce_sum(lrgp._impl_kernel(x, lrgp._x) * U[:, index_j]) 
        
        #def func(x,y):
            #points = tf.constant([x,y], dtype=float_type)
            #out = tf.math.reduce_sum(lrgp._impl_kernel(points, lrgp._x) * U[:, index_i]) * tf.math.reduce_sum(lrgp._impl_kernel(points, lrgp._x) * U[:, index_j]) 
            #return out
            
        def func_ker(x,y):
            point = tf.constant([x,y], dtype=float_type)
            out = lrgp._impl_kernel(point, lrgp._x[index_i]) * lrgp._impl_kernel(point, lrgp._x[index_j])
            return out

        integral = integrate.dblquad( lambda x,y: func_ker(x,y), bounds[0], bounds[1],bounds[0], bounds[1])
        print(integral[0])
        
        #M = tf.random.uniform(shape = (250, 250), dtype=float_type)
        #test = tf.transpose(U) @ M @ U
        
        #a = tf.reshape(tf.transpose(U[:, index_i]) , shape = (1,250))
        #b = tf.reshape(U[:, index_j] , shape = (250,1))
        #test2 = a @ M @ b
        
        trainable = self.process.parameters
        variance = trainable[1]
        lengthscales = tf.square(trainable[0])
        M = tf_calc_Psi_matrix_SqExp(lrgp._x, variance, lengthscales,  domain = lrgp.space.bounds )
    
        out = tf.transpose(U) @ M @ U
        print(M[index_i, index_j])
        

        



if __name__ == '__main__':
    unittest.main()

 
    
    