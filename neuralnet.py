# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:19:14 2019

@author: Tim

rewritten and optimized neural network code of a uni-assignment

Weights of the neurons are stored in numpy arrays for quick access and
optimization
"""

import numpy as np
import math as m


class neuralnet:
    def __init__(self, topology):
        self.inputlayer_size = topology[0]+1
        self.hiddenlayer_size = topology[1]+1
        self.outputlayer_size = topology[2]
        self.create_arrays()
    
    @property
    def learning_rate(self):
        return self.__learning_rate
    
    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate
    
    @learning_rate.getter
    def learning_rate(self):
        return self.__learning_rate
    
    @property
    def inertia(self):
        return self.__inertia
    
    @inertia.setter
    def inertia(self, inertia):
        self.__inertia = inertia
    
    @inertia.getter
    def inertia(self):
        return self.__inertia
    
    @property
    def inputlayer_size(self):
        return self.__inputlayer_size
    
    @inputlayer_size.setter
    def inputlayer_size(self, inputlayer_size):
        self.__inputlayer_size = inputlayer_size
    
    @inputlayer_size.getter
    def inputlayer_size(self):
        return self.__inputlayer_size
    
    @property
    def hiddenlayer_size(self):
        return self.__hiddenlayer_size
    
    @hiddenlayer_size.setter
    def hiddenlayer_size(self, hiddenlayer_size):
        self.__hiddenlayer_size = hiddenlayer_size
    
    @hiddenlayer_size.getter
    def hiddenlayer_size(self):
        return self.__hiddenlayer_size
    
    @property
    def outputlayer_size(self):
        return self.__outputlayer_size
    
    @outputlayer_size.setter
    def outputlayer_size(self, outputlayer_size):
        self.__outputlayer_size = outputlayer_size
    
    @outputlayer_size.getter
    def outputlayer_size(self):
        return self.__outputlayer_size
    
    #Math functions for the neurons:
    def ac_fct(x):
        return ((1)/(1 + m.exp(-x)))
    
    def der_fct(x):
        return (m.exp(-x) / (pow((m.exp(-x) + 1),2)))
    
    #Arrays
    def create_arrays(self):
        
    