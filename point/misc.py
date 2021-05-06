# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:36:21 2021

@author: jesel
"""
import numpy as np



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
    
    
    
if __name__ == "__main__":
    sp = Space([-1,1])
    test = sp.bounds
    print(sp.bounds)

 
    

