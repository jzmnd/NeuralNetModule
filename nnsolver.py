#! /usr/bin/env python
"""
nnsolver.py
Solver class for performing gradient descent training on a model

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

__author__ = "Jeremy Smith"
__version__ = "1.2"


class Solver():
	"""Class containing solver for training of NN or linear classifier models using gradient descent"""
	def __init__(self, model, data_training, data_val, k=1, config="default.config.yml"):
		self.model = model
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.X_train = data_training[0]
		self.y_train = data_training[1]
		self.X_val = data_val[0]
		self.y_val = data_val[1]
		self.k = k
		self.Xinputdims = self.X_train.shape[0]

		self.num_images_train = self.y_train.size
		self.num_images_val = self.y_val.size
		self.count = 1

		self.gradchecks = []
		self.losses = []
		self.accuracies_train = []
		self.accuracies_val = 0

		self.config = load_config(config)

		self.filenametag = "_{:s}_{:.0e}_{:.4f}_{:s}_{:s}_{:s}".format(
			self.model.name,
			self.config['learning_rate'],
			self.config['mu'],
			self.config['btype'],
			self.config['ftype'],
			self.config['update_type'])

		self.update_function = methods[self.config['update_type']]

	def train(self):
		"""Perform training on model"""
		print "TRAINING ON DATA"
		print "  model name         :", self.model.name
		print "  model type         :", self.model.__class__.__name__
		print "  num of points      :", self.num_images_train
		print "  num of dimensions  :", self.Xinputdims
		print "  num of classes     :", self.k
		print "  solver update type :", self.config['update_type'], self.config['btype']
		print "  init learning rate :", self.config['learning_rate']

		self.model.init_weights(self.config['aini'], self.config['bini'], self.num_images_train, self.Xinputdims, self.k)

		results = grad_descent(self.X_train, self.y_train, self.model.weights, self.config, self.model.compute_loss, self.model.forward)

		self.losses = results[0]
		self.model.weights = results[1]
		self.dW = results[2]
		self.gradchecks = results[3]
		self.accuracies_train = results[4]
		self.count = results[5]

		return

	def validate(self):
		"""Check accuracy with validation data"""
		print "VALIDATION ACCURACY"

		return

	def plot(self):
		quickPlot("loss_f{:s}".format(self.filenametag), self.path, [np.arange(self.count-1) + 1, self.losses],
			xlabel="step", ylabel="loss", yscale="log")
		quickPlot("grad_err{:s}".format(self.filenametag), self.path, [np.arange(self.count-1) + 1, self.gradchecks],
			xlabel="step", ylabel="relative grad error", yscale="log")
		quickPlot("accuracy{:s}".format(self.filenametag), self.path, [np.arange(self.count-1) + 1, self.accuracies_train],
			xlabel="step", ylabel="accuracy [%]", yscale="linear")
		return

	def writeout(self):
		dataOutputGen("losses{:s}.txt".format(self.filenametag), self.path, self.losses)
		dataOutputGen("gradcheck{:s}.txt".format(self.filenametag), self.path, self.gradchecks)
		dataOutputGen("accuracy{:s}.txt".format(self.filenametag), self.path, self.accuracies_train)
		return

