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

		# Loads NN architecture and sorts layers by id number
		self.architecture, self.nlayers = load_arch(architecture)
		self.archname = self.architecture['name']
		self.layers = sorted(self.architecture['layers'], key=lambda l: l['id'])

		# Holds cached values for each layer
		self.cache = []

		if not weights:
			initW()

	def initW(self):
		"""Init weights with random numbers"""
		for layer in self.layers:



	def forward(self, X):
		"""Forward propagation of data through NN"""
		for layer in self.layers:
			if layer['type'] == 'inputLayer':
				self.cache.append(X)

			if layer['type'] == 'hiddenLayer':
				activation_function = functions[layer['config']['activation']]
				Wi = self.weights[layer['id'] - 1]

				X = activation_function(np.dot(Wi, X))
				self.cache.append(X)

			if layer['type'] == 'outputLayer':
				Wi = self.weights[layer['id'] - 1]

				X = np.dot(Wi, X)
				self.cache.append(X)

		return X

	def backprop(self, dX):
		"""Backward propagation of gradients through NN"""
		dW_list = []
		for layer in self.layer[::-1]:
			if layer['type'] == 'outputLayer':
				Xi = self.cache[layer['id']]
				Wi = self.weights[layer['id'] - 1]
				
				dX_tmp = np.dot(Wi.T, dX)
				dW = np.dot(dX, Xi.T)

				dX = dX_tmp
				dW_list.append(dW)

			if layer['type'] == 'hiddenLayer':
				activation_function_grad = gradientfunctions[layer['config']['activation']]
				Xi = self.cache[layer['id']]
				Wi = self.weights[layer['id'] - 1]

				dX = activation_function_grad(Xi) * dX

				dX_tmp = np.dot(Wi.T, dX)
				dW = np.dot(dX, Xi.T)

				dX = dX_tmp
				dW_list.append(dW)

			if layer['type'] == 'inputLayer':
				continue

		return dW_list[::-1]


	def compute_loss(self):
		return


