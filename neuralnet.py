#! /usr/bin/env python
"""
neuralnet.py
Neural Network Classes

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

import os
import sys
from myfunctions import *
import numpy as np
from lossfunctionlib import *
from graddescentlib import *
from datautils import *
from funcutils import *
import yaml

__author__ = "Jeremy Smith"
__version__ = "1.0"


class FeedForwardNN():
	"""Class for descibing basic feed forward neural network"""
	def __init__(self, weights=None, architecture="default.arch.yml", name="default"):
		self.weights = weights
		self.name = name
		self.path = os.path.dirname(os.path.abspath(__file__))

		self.architecture, self.nlayers = load_arch(architecture)
		self.archname = self.architecture['name']
		self.layers = sorted(self.architecture['layers'], key=lambda l: l['id'])

	def forward(self, X, W):
		for layer in self.layers:
			if layer['type'] == 'inputLayer':
				continue
			if layer['type'] == 'hiddenLayer':
				acticvation_function = functions[layer['config']['activation']]
				Wi = W[layer['id'] - 1]
				X = activation_fuction(np.dot(Wi, X))
			if layer['type'] == 'outputLayer':
				Wi = W[layer['id'] - 1]
				X = np.dot(Wi, X)
		return X


	def backprop(self, X, W):
		for layer in self.layer[::-1]:
			if layer['type'] == 'outputLayer':
				
		return


	def compute_loss(self):
		return

	def param_update(self):
		return



	def run(self):
		return



	def plot(self):
		return



	def writeout(self):
		return






# class RecursiveNN():
# 	"""Class for descibing recursive neural network"""
# 	def __init__(self, weights, bias):


# class ConvNN():
# 	"""Class for descibing convolutional neural network"""
# 	def __init__(self, weights, bias):
