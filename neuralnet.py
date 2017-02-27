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
	"""Class for descibing basic fully connected feed forward neural network"""
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

	def forward(self, X, W=None):
		"""Forward propagation of data through NN"""
		if W is None:
			W = self.weights
		if W is None:
			print "Train NN or provide explicit weights"
		# Clear cache
		self.cache = []
		for layer in self.layers:
			if layer['type'] == 'inputLayer':
				# Append output to cache
				self.cache.append(X)

			if layer['type'] == 'hiddenLayer':
				# Find weights and activation function for layer
				activation_function = functions[layer['config']['activation']]
				Wi = W[layer['id'] - 1]
				# Calculate output of layer and append to cache
				X = activation_function(np.dot(Wi, X))
				self.cache.append(X)

			if layer['type'] == 'outputLayer':
				# Find weights for layer
				Wi = W[layer['id'] - 1]
				# Calculate output of layer and append to cache
				X = np.dot(Wi, X)
				self.cache.append(X)
		return X

	def backprop(self, dX, W=None):
		"""Backward propagation of gradients through NN"""
		if W is None:
			W = self.weights
		if W is None:
			print "Train NN or provide explicit weights"
		dW_list = []
		for layer in self.layers[::-1]:
			if layer['type'] == 'outputLayer':
				# Find weights and cached output for layer
				Xi1 = self.cache[layer['id'] - 1]
				Wi = W[layer['id'] - 1]
				# Calculate gradients and append to dw list
				dX_tmp = np.dot(Wi.T, dX)
				dW = np.dot(dX, Xi1.T)
				dX = dX_tmp
				dW_list.append(dW)

			if layer['type'] == 'hiddenLayer':
				# Find weights, cached output and activation function for layer
				activation_function_grad = gradientfunctions[layer['config']['activation']]
				Xi = self.cache[layer['id']]
				Xi1 = self.cache[layer['id'] - 1]
				Wi = W[layer['id'] - 1]
				# Calculate gradients and append to dW list
				dX = np.multiply(activation_function_grad(Xi), dX)
				dX_tmp = np.dot(Wi.T, dX)
				dW = np.dot(dX, Xi1.T)
				dX = dX_tmp
				dW_list.append(dW)

			if layer['type'] == 'inputLayer':
				continue
		return np.array(dW_list[::-1])

	def compute_loss(self, X, y, W):
		"""Computes loss and gradients wrt weights, based on scores compared to y"""
		# Compute forward pass scores
		scores = self.forward(X, W=W)

		# Loss function type (config on last layer)
		ftype = self.layers[-1]['config']['ftype']
		# Calculate loss and gradient on loss wrt weights
		loss, dscores = loss_function(scores, y, ftype=ftype)

		# Backprop of gradient
		dW = self.backprop(dscores, W=W)

		# Add regularization loss to each layer
		for layer in self.layers:
			if layer['type'] == 'inputLayer':
				continue
			else:
				if layer['config']['reg']:
					loss += 0.5 * layer['config']['gamma'] * np.sum(W[layer['id'] - 1]**2)
					dW[layer['id'] - 1] += layer['config']['gamma'] * W[layer['id'] - 1]
		return loss, dW
