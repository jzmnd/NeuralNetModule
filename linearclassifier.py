#! /usr/bin/env python
"""
linearclassifier.py
Linear Classifier Class

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

__author__ = "Jeremy Smith"
__version__ = "1.0"


class LinearClassifier():
	"""Class for general linear classifier"""
	def __init__(self, config="default.config.yml", weights=None, name="default"):
		self.weights = weights
		self.name = name
		self.path = os.path.dirname(os.path.abspath(__file__))

		self.config = load_config(config)

		self.filenametag = "_{:s}_{:.0e}_{:.4f}_{:.4f}_{:s}_{:s}_{:s}".format(
			name, self.config['learning_rate'], self.config['gamma'], self.config['mu'], self.config['btype'], self.config['ftype'], self.config['update_type'])

		self.update_function = methods[self.config['update_type']]

	def forward(self, X, W=None):
		if W is None:
			W = self.weights
		if W is None:
			print "Train classifier or provide explicit weights"
		return score_function(X, W)

	def train(self, X, y, k=1):
		print "TRAINING ON DATA"
		print "  num of points      :", y.size
		print "  num of dimensions  :", X.shape[0]
		print "  num of classes     :", k

		X = biastrick(X)
		Winit = initializeweights(y.size, X.shape[0], k)

		results = grad_descent(X, y, Winit, self.config, self.compute_loss, self.forward)

		self.losses = results[0]
		self.weights = results[1]
		self.dW = results[2]
		self.gradcheck = results[3]
		self.accuracy = results[4]
		self.count = results[5]

		return

	def compute_loss(self, X, y, W):
		scores = score_function(X, W)
		loss, dscores = loss_function(scores, y, ftype=self.config['ftype'])
		dW = np.dot(dscores, X.T)

		if self.config['reg']:
			loss += 0.5 * self.config['gamma'] * np.sum(W**2)
			dW += self.config['gamma'] * W

		return loss, dW

	def plot(self):
		quickPlot("loss_f{:s}".format(self.filenametag), self.path, [np.arange(self.count-1) + 1, self.losses],
			xlabel="step", ylabel="loss", yscale="log")
		quickPlot("grad_err{:s}".format(self.filenametag), self.path, [np.arange(self.count-1) + 1, self.gradcheck],
			xlabel="step", ylabel="relative grad error", yscale="log")
		quickPlot("accuracy{:s}".format(self.filenametag), self.path, [np.arange(self.count-1) + 1, self.accuracy],
			xlabel="step", ylabel="accuracy [%]", yscale="linear")
		return

	def writeout(self):
		dataOutputGen("weights{:s}.txt".format(self.filenametag), self.path, self.weights)
		dataOutputGen("losses{:s}.txt".format(self.filenametag), self.path, self.losses)
		dataOutputGen("gradcheck{:s}.txt".format(self.filenametag), self.path, self.gradcheck)
		dataOutputGen("accuracy{:s}.txt".format(self.filenametag), self.path, self.accuracy)
		return
