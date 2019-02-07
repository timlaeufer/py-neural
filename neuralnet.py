# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:19:14 2019

@author: Tim

rewritten and optimized neural network code of a uni-assignment

Weights of the neurons are stored in numpy arrays for quick access and
optimization

 * <topology>
 *
 * <weights of input neuron 0>
 * <weights of input neuron 1>
 * <...>
 * [blank line]
 * <weights of hidden neuron 0>
 * <weights of hidden neuron 1>
 * [end, no blank line]
 *
"""

import numpy as np
import math as m
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


class neuralnet:
    def __init__(self, topology):
        self.bias = 1
        self.learning_rate = 0.9
        self.alpha = 0.01
        self.inputlayer_size = topology[0]+1
        self.hiddenlayer_size = topology[1]+1
        self.outputlayer_size = topology[2]
        self.create_arrays()
        #self.print_net()
    
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
    
    @property
    def inputlayer_input(self):
        return self.__inputlayer_input
    
    @inputlayer_input.setter
    def inputlayer_input(self, inputlayer_input):
        self.__inputlayer_input = inputlayer_input
    
    @inputlayer_input.getter
    def inputlayer_input(self):
        return self.__inputlayer_input
    
    @property
    def hiddenlayer_input(self):
        return self.__hiddenlayer_input
    
    @property
    def inputlayer_output(self):
        return self.__inputlayer_output
    
    @inputlayer_output.setter
    def inputlayer_output(self, inputlayer_output):
        self.__inputlayer_output = inputlayer_output
    
    @inputlayer_output.getter
    def inputlayer_output(self):
        return self.__inputlayer_output
    
    @hiddenlayer_input.setter
    def hiddenlayer_input(self, hiddenlayer_input):
        self.__hiddenlayer_input = hiddenlayer_input
    
    @hiddenlayer_input.getter
    def hiddenlayer_input(self):
        return self.__hiddenlayer_input
    
    @property
    def hiddenlayer_output(self):
        return self.__hiddenlayer_output
    
    @hiddenlayer_output.setter
    def hiddenlayer_output(self, hiddenlayer_output):
        self.__hiddenlayer_output = hiddenlayer_output
    
    @hiddenlayer_output.getter
    def hiddenlayer_output(self):
        return self.__hiddenlayer_output
    
    @property
    def outputlayer_input(self):
        return self.__outputlayer_input
    
    @outputlayer_input.setter
    def outputlayer_input(self, outputlayer_input):
        self.__outputlayer_input = outputlayer_input
    
    @outputlayer_input.getter
    def outputlayer_input(self):
        return self.__outputlayer_input
    
    @property
    def outputlayer_output(self):
        return self.__outputlayer_output
    
    @outputlayer_output.setter
    def outputlayer_output(self, outputlayer_output):
        self.__outputlayer_output = outputlayer_output
    
    @outputlayer_output.getter
    def outputlayer_output(self):
        return self.__outputlayer_output
    
    @property
    def inputlayer_weights(self):
        return self.__inputlayer_weights
    
    @inputlayer_weights.setter
    def inputlayer_weights(self, inputlayer_weights):
        self.__inputlayer_weights = inputlayer_weights
    
    @inputlayer_weights.getter
    def inputlayer_weights(self):
        return self.__inputlayer_weights
    
    @property
    def hiddenlayer_weights(self):
        return self.__hiddenlayer_weights
    
    @hiddenlayer_weights.setter
    def hiddenlayer_weights(self, hiddenlayer_weights):
        self.__hiddenlayer_weights = hiddenlayer_weights
    
    @hiddenlayer_weights.getter
    def hiddenlayer_weights(self):
        return self.__hiddenlayer_weights
    
    @property
    def old_inputlayer_weights(self):
        return self.__old_inputlayer_weights
    
    @old_inputlayer_weights.setter
    def old_inputlayer_weights(self, old_inputlayer_weights):
        self.__old_inputlayer_weights = old_inputlayer_weights
    
    @old_inputlayer_weights.getter
    def old_inputlayer_weights(self):
        return self.__old_inputlayer_weights
    
    @property
    def old_hiddenlayer_weights(self):
        return self.__old_hiddenlayer_weights
    
    @old_hiddenlayer_weights.setter
    def old_hiddenlayer_weights(self, old_hiddenlayer_weights):
        self.__old_hiddenlayer_weights = old_hiddenlayer_weights
    
    @old_hiddenlayer_weights.getter
    def old_hiddenlayer_weights(self):
        return self.__old_hiddenlayer_weights
    
    @property
    def bias(self):
        return self.__bias
    
    @bias.setter
    def bias(self, bias):
        self.__bias = bias
    
    @bias.getter
    def bias(self):
        return self.__bias
    
    @property
    def alpha(self):
        return self.__alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self.__alpha = alpha
    
    @alpha.getter
    def alpha(self):
        return self.__alpha
    
    @property
    def images(self):
        return self.__images
    
    @images.setter
    def images(self, images):
        self.__images = images
    
    @images.getter
    def images(self):
        return self.__images
    
    @property
    def labels(self):
        return self.__labels
    
    @labels.setter
    def labels(self, labels):
        self.__labels = labels
    
    @labels.getter
    def labels(self):
        return self.__labels
        
    #Math functions for the neurons:
    def ac_fct(self, x):
        return ((1)/(1 + m.exp(-x)))
    
    def der_fct(self, x):
        return (m.exp(-x) / (pow((m.exp(-x) + 1),2)))
    
    #Arrays
    def create_arrays(self):

        self.inputlayer_input = np.zeros((self.inputlayer_size))
        self.inputlayer_weights = np.random.standard_normal((
                                            self.inputlayer_size, 
                                            self.hiddenlayer_size))
        self.inputlayer_output = np.ones((self.inputlayer_size))
        self.old_inputlayer_weights = np.zeros((
                                            self.inputlayer_size, 
                                            self.hiddenlayer_size))
        self.hiddenlayer_input = np.ones((self.hiddenlayer_size))
        self.hiddenlayer_weights = np.random.standard_normal((
                                             self.hiddenlayer_size,
                                             self.outputlayer_size))
        self.hiddenlayer_output = np.ones((self.hiddenlayer_size))
        self.old_hiddenlayer_weights = np.zeros((
                                            self.hiddenlayer_size, 
                                            self.outputlayer_size))
        self.outputlayer_input = np.zeros((self.outputlayer_size))
        self.outputlayer_output = np.zeros((self.outputlayer_size))
        
        self.inputlayer_output[0] = self.bias
        self.hiddenlayer_output[0] = self.bias
        
    def print_net(self):
        print(self.inputlayer_size, self.hiddenlayer_size,
              self.outputlayer_size)
        print(self.inputlayer_weights, "\n")
        print(self.hiddenlayer_weights)

    def set_weight(self, layer, neuron, target, new_weight):
        if(layer == 0):
            self.inpulayer_weights[neuron, target] = new_weight
        elif(layer == 1):
            self.hiddenlayer_weights[neuron, target] = new_weight
            
    def calc_output(self, data):
        if(len(data) != self.inputlayer_size-1):
            print("DATA DOESNT HAVE THE CORRECT LENGTH!")
            return
        for i, el in enumerate(data):
            self.inputlayer_output[i] = (el/255)
        self.inputlayer_output[-1] = 1
        
        #hidden layer:
        for i in range(self.hiddenlayer_size):
            temp = 0
            for j in range(self.inputlayer_size):
                temp += self.inputlayer_output[j]*self.inputlayer_weights[j,i]
            self.hiddenlayer_input[i] = temp
            self.hiddenlayer_output[i] = self.ac_fct(temp)
            
        self.hiddenlayer_output[0] = self.bias
            
        #output layer:
        for i in range(self.outputlayer_size):
            temp = 0
            for j in range(self.hiddenlayer_size):
                temp += self.hiddenlayer_output[j] * self.hiddenlayer_weights[j,i]
            self.outputlayer_input[i] = temp
            self.outputlayer_output = self.ac_fct(temp)
            
        return self.outputlayer_output
            
            
    def get_net_output(self):
        return self.outputlayer_output
            
    def back(self, expected):
        for i in range(self.outputlayer_size):
            for j in range(self.hiddenlayer_size):
                newD = self.calc_output_gradient(i, j, expected[i])
                newD_in = (1-self.alpha) * newD
                newD_in += self.alpha*self.old_hiddenlayer_weights[j, i]
                
                new_w = self.hiddenlayer_weights[j, i] + newD_in
                self.old_hiddenlayer_weights[j, i] = self.hiddenlayer_weights[j, i]
                self.hiddenlayer_weights[j, i] = new_w
        
        for i in range(self.hiddenlayer_size):
            for j in range(self.inputlayer_size):
                newD = self.calc_hidden_gradient(i, j, expected)
                newD_in = (1-self.alpha)*newD + self.alpha*self.old_inputlayer_weights[j, i]
                new_w = self.inputlayer_weights[j, i] + newD_in
                self.old_inputlayer_weights[j, i] = self.inputlayer_weights[j, i]
                self.inputlayer_weights[j, i] = new_w
    
    def calc_output_gradient(self, i_out, j_hid, expected):
        a = self.ac_fct(self.hiddenlayer_input[j_hid] - self.bias)
        my_inp = self.outputlayer_input[i_out]
        v = 2*self.learning_rate*(expected-self.ac_fct(my_inp))*self.der_fct(
                my_inp) * a
        return v
    
    def calc_hidden_gradient(self, i_hid, j_inp, expect_arr):
        s = 0.0
        for k in range(self.outputlayer_size):
            ex = expect_arr[k]
            s+=ex - self.ac_fct(self.outputlayer_input[k])
            s*=self.der_fct(self.outputlayer_input[k])
            s*=self.hiddenlayer_weights[i_hid, k]
            
        a = self.ac_fct(self.inputlayer_input[j_inp] - self.bias)
        return (2*self.learning_rate*s*self.der_fct(
                self.hiddenlayer_input[i_hid])*a)
        
    def save_net(self, filename):
        with open(filename) as f:
            f.write(""+(self.inputlayer_size-1)+" "+(self.hiddenlayer_size-1)+
                    " "+self.outputlayer_size + "\n\n")
            for i in range(self.inputlayer_size):
                for j in range(self.hiddenlayer_size):
                    f.write(self.inputlayer_weights[i, j] + " ")
                f.write("\n")
            for i in range(self.hiddenlayer_weights):
                for j in range(self.outputlayer_size):
                    f.write(self.hiddenlayer_weights[i, j] + " ")
                f.write("\n")
                
    def load_binaries(self, bin_name, label_name, show=False):
        bin_dat = ""
        label_dat = ""
        
        with open(bin_name, mode = 'rb') as f:
            bin_dat = f.read()
        with open(label_name, mode = 'rb') as f:
            label_dat = f.read()
            
        bin_int = [x for x in bin_dat]
        label_int = [x for x in label_dat]
        self.images = []
        tempimg = []
        counter = 1
        print("Loading binaries ...")
        for i in tqdm(range(16, len(bin_int))):
            tempimg.append(bin_int[i])
            if(counter == 784):
                counter = 1
                self.images.append(tempimg)
                tempimg = []
            else:
                counter += 1
                
        self.labels = label_int[8:]
        print("Binaries loaded!")
        
        self.loaded = True
        if not show: 
            return
        
        for i in range(len(self.images)):
            print(self.labels[i], i)
            arr = np.array(self.images[i])
            img = np.reshape(arr, (28, 28))
            plt.imshow(img)
            plt.show()
            time.sleep(5)
        
        
    def train_net(self):
        """trains the net with the given binaries"""
        if not self.loaded:
            print("Load me first!")
            return
        
        print("Training the net ...")
        for i in tqdm(range(len(self.images))):
            expected = [0 for x in range(10)]
            expected[self.labels[i]] = 1
            self.calc_output(self.images[i])
            self.back(expected)
        
        self.save_net("saved_net_"+time.localtime+".txt")
        
            
x = neuralnet([784, 64, 10])
x.load_binaries("images.bin", "images.labels")
x.train_net()