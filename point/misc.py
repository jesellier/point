# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np



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