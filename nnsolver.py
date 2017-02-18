#! /usr/bin/env python
"""
nnsolver.py
Solver class for performing gradient descent training on a NN model

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


class Solver():
	"""Class containing solver for training of NN using gradient descent"""
	def __init__(self, model, data_training, data_val, k=1, config="default.config.yml"):
		self.model = model
		self.X_train = data_training[0]
		self.y_train = data_training[1]
		self.X_val = data_val[0]
		self.y_val = data_val[1]
		self.k = k

		self.num_images_train = self.y_train.size
		self.num_images_val = self.y_val.size
		self.count = 1

		self.gradchecks = []
		self.losses = []
		self.accuracies_train = []
		self.accuracies_val = []

		self.config = load_config(config)

		self.filenametag = "_{:s}_{:.0e}_{:.4f}_{:s}_{:s}".format(
			self.model.name,
			self.config['learning_rate'],
			self.config['mu'],
			self.config['btype'],
			self.config['update_type'])

		self.update_function = methods[self.config['update_type']]

	def train(self):
		print "TRAINING ON DATA"
		print "  model name         :", self.model.name
		print "  num of points      :", self.num_images_train
		print "  num of dimensions  :", self.X_train.shape[0]
		print "  num of classes     :", self.k
		print "  solver update type :", self.config['update_type'], self.config['btype']
		print "  init learning rate :", self.config['learning_rate']

		self.init_weights(self.config['aini'], self.config['bini'])

		# TODO: perform grad descent and updates

		return

	def init_weights(self, a, b):
		if not self.model.weights:
			self.model.weights = []
			for layer in self.model.layers:
				if layer['type'] == 'inputLayer':
					d = layer['config']['dims']
				else:
					self.model.weights.append(initializeweights(self.num_images_train, d, k=layer['config']['dims'], a=a, b=b))
					d = layer['config']['dims']
		return
