# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import scipy.optimize
import tensorflow as tf

Variables = Iterable[tf.Variable] 
LossClosure = Callable[[], tf.Tensor]



class Space():
    def __init__(self, bound = [-1,1]):
        self._lower_bound =  bound[0]
        self._higher_bound = bound[1]
        
    
    @property
    def bounds(self):
        return self.bounds2D
    
    @property
    def bounds2D(self):
        return np.array([[self._lower_bound,   self._higher_bound ], [ self._lower_bound,  self._higher_bound ]]) 
    
    @property
    def bounds1D(self):
        return np.array((self._lower_bound,  self._higher_bound )) 
        
    @property
    def measure(self):
        return (self.__x2Max() - self.__x1Min()) * (self.__x2Max() - self.__x1Min())
    
    @property
    def center(self):
        return [(self.__x1Min() + self.__x1Max())/2, (self.__x2Min() + self.__x2Max())/2]
 
    def __x1Min(self):
        return self._lower_bound
    
    def __x1Max(self):
        return self._higher_bound

    def __x2Min(self):
        return self._lower_bound

    def __x2Max(self):
        return self._higher_bound 
    
    
    
class TensorMisc():
    
    @staticmethod
    def pack_tensors(tensors: Sequence[Union[tf.Tensor, tf.Variable]]) -> tf.Tensor:
        flats = [tf.reshape(tensor, (-1,)) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector
    
    
    @staticmethod
    def pack_tensors_to_zeros(tensors: Sequence[Union[tf.Tensor, tf.Variable]]) -> tf.Tensor:
        flats = [tf.zeros(shape = tf.reshape(tensor, (-1,)).shape, dtype = tensor.dtype) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector


    @staticmethod
    def unpack_tensors(
        to_tensors: Sequence[Union[tf.Tensor, tf.Variable]], from_vector: tf.Tensor) -> List[tf.Tensor]:
        s = 0
        values = []
        for target_tensor in to_tensors:
            shape = tf.shape(target_tensor)
            dtype = target_tensor.dtype
            tensor_size = tf.reduce_prod(shape)
            tensor_vector = from_vector[s : s + tensor_size]
            tensor = tf.reshape(tf.cast(tensor_vector, dtype), shape)
            values.append(tensor)
            s += tensor_size
        return values
    

    @staticmethod
    def assign_tensors(to_tensors: Sequence[tf.Variable], values: Sequence[tf.Tensor]) -> None:
        if len(to_tensors) != len(values):
            raise ValueError("to_tensors and values should have same length")
        for target, value in zip(to_tensors, values):
            target.assign(value)
    
   

 
    

