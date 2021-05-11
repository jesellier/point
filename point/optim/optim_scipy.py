# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:32:49 2021

@author: jesel
"""

import numpy as np
import scipy.optimize
import tensorflow as tf

from point.misc import TensorMisc



def initial_parameters(variables) -> tf.Tensor:
     return TensorMisc().pack_tensors(variables)



class OptimScipy() :
    
    def minimize(self, closure, variables, method = "L-BFGS-B", compile = False, **scipy_kwargs):

        if not callable(closure):
            raise TypeError(
                "The 'closure' argument is expected to be a callable object."
            )  # pragma: no cover
        variables = tuple(variables)
        if not all(isinstance(v, tf.Variable) for v in variables):
            raise TypeError(
                "The 'variables' argument is expected to only contain tf.Variable instances (use model.trainable_variables, not model.trainable_parameters)"
            )  # pragma: no cover
        initial_params = initial_parameters(variables)

        func = self.eval_func(closure, variables, compile=compile)

        return scipy.optimize.minimize(
            func, initial_params, jac=True, method=method, **scipy_kwargs
        )
    
    @classmethod
    def eval_func(cls, closure, variables, compile: bool = False) :
        def _tf_eval(x: tf.Tensor):
            values = TensorMisc().unpack_tensors(variables, x)
            TensorMisc().assign_tensors(variables, values)
            loss, grads = closure()
            return loss,  TensorMisc().pack_tensors(grads)

        if compile:
            _tf_eval = tf.function(_tf_eval)

        def _eval(x: np.ndarray) :
            loss, grad = _tf_eval(tf.convert_to_tensor(x))
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        return _eval