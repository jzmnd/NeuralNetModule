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
import numpy as np
from lossfunctionlib import *
from utils.datautils import *
from utils.funcutils import *

__author__ = "Jeremy Smith"
__version__ = "1.1"


class LinearClassifier():
	"""Class for general linear classifier"""
	def __init__(self, weights=None, config="default.config.yml", name="default"):
		self.weights = weights
		self.name = name
		self.path = os.path.dirname(os.path.abspath(__file__))

		self.config = load_config(config)

	def forward(self, X, W=None):
		"""Forward pass through linear classifier"""
		if W is None:
			W = self.weights
		if W is None:
			print "ERROR: Train classifier or provide explicit weights"
		return score_function(X, W)

	def compute_loss(self, X, y, W):
		"""Computes loss and gradients wrt weights, based on scores compared to y"""
		scores = score_function(X, W)
		loss, dscores = loss_function(scores, y, ftype=self.config['ftype'])
		dW = np.dot(dscores, X.T)

		if self.config['reg']:
			loss += 0.5 * self.config['gamma'] * np.sum(W**2)
			dW += self.config['gamma'] * W

		return loss, dW

	def init_weights(self, a, b, n, d, k):
		"""Initialize weights if not already initialized"""
		if self.weights is None:
			print "INITIALIZING WEIGHTS..."
			self.weights = initializeweights(n, d, k=k, a=a, b=b)
		return

