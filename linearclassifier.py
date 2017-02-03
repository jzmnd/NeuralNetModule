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

	def train(self, X, y, k=1):
		print "TRAINING ON DATA"
		print "  num of points      :", y.size
		print "  num of dimensions  :", X.shape[0]
		print "  num of classes     :", k

		X = biastrick(X)
		Winit = initializeweights(X, k)

		results = grad_descent(X, y, Winit, self.config)

		self.losses = results[0]
		self.weights = results[1]
		self.dW = results[2]
		self.gradcheck = results[3]
		self.accuracy = results[4]
		self.count = results[5]

		return

	def run(self, X):
		if not weights:
			print "Train classifier first!"
			return
		return score_function(X, self.weights)

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
