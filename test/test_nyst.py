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

import time
from point.point_process import Space

import unittest


from scipy.optimize import minimize #For optimizing
from point.helper import method, get_process


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
    


class Test_Nystrom(unittest.TestCase):
     
    #def setUp(self):
    def setUp(self):
        rng = np.random.RandomState()
        self.variance = tf.Variable(8, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')

        self.process = get_process(self.length_scale, self.variance, method = method.NYST, n_components = 250, random_state = rng)


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
        



if __name__ == '__main__':
    unittest.main()

 
    
    