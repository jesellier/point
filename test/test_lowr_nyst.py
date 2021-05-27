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

    erf_val = tf.math.erf((zm - Tmin) * inv_lengthscales) - tf.math.erf((zm - Tmax) * inv_lengthscales)
    product = tf.reduce_prod(mult * erf_val, axis=2)
    out = tf.square(variance) * tf.exp(exp_arg + tf.math.log(product))
    return out


def tf_calc_Psi_vector_SqExp(z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z) = ∫ K(z,x) dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).
    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.
    Does not broadcast over leading dimensions.
    """
    
    variance = tf.cast(variance, z.dtype)
    lengthscales = tf.cast(lengthscales, z.dtype)

    mult = tf.cast(np.sqrt(0.5 * np.pi), z.dtype) * lengthscales
    inv_lengthscales = 1.0 / lengthscales

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]
    
    erf_val = tf.math.erf(np.sqrt(0.5) * (z - Tmin) * inv_lengthscales) - tf.math.erf(np.sqrt(0.5) * (z - Tmax) * inv_lengthscales)
    product = tf.reduce_prod(mult * erf_val, axis=1)
    out = variance * product

    return out
    


class Test_Nystrom(unittest.TestCase):
     
    #def setUp(self):
    def setUp(self):
        self.rng = np.random.RandomState(5)
        self.variance = tf.Variable(1, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([1,1], dtype=float_type, name='l')
        self.space = Space([-1,1])


    def test_singularity(self):
        process = get_process(length_scale = self.length_scale, variance = self.variance, 
                                   space = self.space, method = method.NYST, n_components = 250, random_state = self.rng)
        s = process.lrgp._lambda.numpy()

        eps = eigvalsh_to_eps(s, None, None)
        d = s[s > eps]
        
        # if np.min(s) < -eps:
        #     raise ValueError('the input matrix must be positive semidefinite')
        # if len(d) < len(s) :
        #     raise np.linalg.LinAlgError('singular matrix')

        self.assertTrue(not (np.min(s) < -eps))  #the input matrix must be positive semidefinite
        self.assertTrue(not (len(d) < len(s)))   #the input matrix must be singular
        

    def test_func_recalculation(self):
        
        process = get_process(length_scale = self.length_scale, variance = self.variance, 
                                   space = self.space, method = method.NYST, n_components = 250, random_state = self.rng)
        lrgp = process.lrgp
        X = tf.constant(rng.normal(size = [10, 2]), dtype=float_type, name='X')

        f = lrgp.func(X).numpy()
        feature = lrgp.feature(X)
        w = tf.linalg.diag(tf.math.sqrt(lrgp._lambda)) @ lrgp._latent
        f_recalc = feature @ w
        f_recalc = f_recalc.numpy()

        for i in range(f.shape[0]):
            self.assertAlmostEqual(f[i][0], f_recalc[i][0], places=7)
            
 
    def test_prod_kernel_integral(self):
        
        variance = self.variance
        length_scale = self.length_scale

        process = get_process(length_scale = length_scale, variance = variance, 
                                   space = self.space, method = method.NYST, n_components = 2, random_state = self.rng)
        lrgp = process.lrgp
        bounds = lrgp.space.bounds1D

        def integral_ker(index_i, index_j) :
            
            def func_ker(x,y):
                point = tf.constant([x,y], dtype=float_type)
                out = lrgp.kernel(point, lrgp._x[index_i]) * lrgp.kernel(point, lrgp._x[index_j])
                return out
    
            integral_ker = integrate.dblquad( lambda x,y: func_ker(x,y), bounds[0], bounds[1],bounds[0], bounds[1])
            return integral_ker[0]

        M = tf_calc_Psi_matrix_SqExp(lrgp._x, variance, length_scale,  domain = lrgp.space.bounds )
        self.assertAlmostEqual(M[0,0].numpy(), integral_ker(0, 0) , places=7)
        self.assertAlmostEqual(M[0,1].numpy(), integral_ker(0, 1) , places=7)
        self.assertAlmostEqual(M[1,1].numpy(), integral_ker(1, 1) , places=7)
        
        
    def test_kernel_integral(self):
        
        variance = tf.Variable(3, dtype=float_type, name='sig')
        length_scale = tf.Variable([2,0.5], dtype=float_type, name='l') #self.length_scale
        process = get_process(length_scale = length_scale, variance = variance, 
                                   space = self.space, method = method.NYST, n_components = 2, random_state = self.rng)
        lrgp = process.lrgp
        bounds = lrgp.space.bounds1D

        def integral_ker(index_i) :
            
            def func_ker(x,y):
                point = tf.constant([x,y], dtype=float_type)
                out = lrgp.kernel(point, lrgp._x[index_i])
                return out
    
            integral_ker = integrate.dblquad( lambda x,y: func_ker(x,y), bounds[0], bounds[1], bounds[0], bounds[1])
            return integral_ker[0]

        phi = tf_calc_Psi_vector_SqExp(lrgp._x, variance, length_scale,  domain = lrgp.space.bounds )

        self.assertAlmostEqual(phi[0].numpy(), integral_ker(0) , places=7)
        self.assertAlmostEqual(phi[1].numpy(), integral_ker(1) , places=7)


    def test_full_integral(self):
        
        variance = self.variance
        length_scale = self.length_scale

        process = get_process(length_scale = length_scale, variance = variance, 
                                   space = self.space, method = method.NYST, n_components = 250, random_state = self.rng)
        lrgp = process.lrgp
        bounds = lrgp.space.bounds1D

        def func(x,y):
            points = tf.constant([x,y], dtype=float_type)
            out = lrgp.func(points)**2
            return out

        M = tf_calc_Psi_matrix_SqExp(lrgp._x, variance, length_scale,  domain = lrgp.space.bounds )
        
        v = lrgp._v
        out = tf.transpose(v) @ M @ v
        out = out.numpy()[0][0]
        
        integral_recalc = integrate.dblquad( lambda x,y: func(x,y), bounds[0], bounds[1],bounds[0], bounds[1])
        self.assertAlmostEqual(out, integral_recalc[0]  , places=7)
        
        
        
    def test_full_integral_with_beta(self):
        
        variance = self.variance
        length_scale = self.length_scale
        beta0 = 0.5

        process = get_process(beta0 = beta0, length_scale = length_scale, variance = variance, 
                                   space = self.space, method = method.NYST, n_components = 250, random_state = self.rng)
        lrgp = process.lrgp
        bounds = lrgp.space.bounds1D

        def func(x,y):
            points = tf.constant([x,y], dtype=float_type)
            out = (lrgp.func(points) + beta0)**2
            return out
        
        integral = lrgp.integral().numpy()
        integral_recalc = integrate.dblquad( lambda x,y: func(x,y), bounds[0], bounds[1],bounds[0], bounds[1])
        self.assertAlmostEqual(integral, integral_recalc[0]  , places=7)

 
if __name__ == '__main__':
    unittest.main()

 
    
    