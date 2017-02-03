#! /usr/bin/env python
"""
lossfunctionlib.py
Loss Function Library Module

 -- Contains loss function for SVM and Softmax

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

from __future__ import division
import os
import sys
from myfunctions import *
import numpy as np
import random as rnd

__author__ = "Jeremy Smith"
__version__ = "1.1"


def score_function(x, W):
	"""Simple linear score function"""
	return np.dot(W, x)


def score_function_bias(x, W, b):
	"""Simple linear score function with explicit bias"""
	return W.dot(x) + b


def loss_function(X, y, W, sf=score_function, ftype='svm', delta=1.0, reg=True, gamma=0.01):
	"""
	Loss function implementation (fully vectorized)
	-- X: holds all the data as columns (e.g. 3073 x 50,000 in CIFAR-10)
	-- y: array of integers specifying correct class (e.g. 50,000-D array)
	-- W: weights (e.g. 10 x 3073)
	-- sf: the score function to be implemented
	-- type: can be either 'svm' or 'softmax'
	-- reg: sets whether to include regularization loss
	-- gamma: regularization hyperparameter
	-- Also works if number of images is 1
	"""

	num_classes = W.shape[0]      # K
	num_images = y.size           # N
	num_pixels = X.shape[0] - 1   # D, allows for final 1 (from bias trick)

	scores = sf(X, W)

	grad = np.zeros(W.shape)

	if ftype == 'svm':
		if num_images == 1:
			# compute the margins for all classes and images
			margins = np.maximum(0, scores - scores[y] + delta)
			# ignore the y-th position and only consider margin on wrong classes
			margins[y] = 0
		else:
			# compute the margins for all classes and images
			margins = np.maximum(0, scores - scores[y, np.arange(num_images)] + delta)
			# ignore the y-th position and only consider margin on wrong classes
			margins[y, np.arange(num_images)] = 0

		# calculte loss (averaged over all images)
		loss = np.sum(margins) / num_images

		# calculate the indicator function for all classes and images
		indicatorf = np.where(margins > 0, 1, 0)
		# set y-th positions to be the sum of indicator values for each image
		if num_images == 1:
			indicatorf[y] = -np.sum(indicatorf, axis=0)
		else:
			indicatorf[y, np.arange(num_images)] = -np.sum(indicatorf, axis=0)

		# calculte gradient (averaged over all images)
		grad = np.dot(indicatorf, X.T) / num_images

	elif ftype == 'softmax':
		# first shift scores so highest value is 0
		scores_shift = scores - np.max(scores)
		exp_scores = np.exp(scores_shift)
		# compute the cross entropy loss for each image
		if num_images == 1:
			loss_i = -scores_shift[y] + np.log(np.sum(exp_scores, axis=0) + 1e-150)
		else:
			loss_i = -scores_shift[y, np.arange(num_images)] + np.log(np.sum(exp_scores, axis=0) + 1e-150)

		# calculte loss (averaged over all images)
		loss = np.sum(loss_i) / num_images

		# calculate coefficient for gradient for each image (allow for zero denominator)
		probs = exp_scores * (1.0 / (np.sum(exp_scores, axis=0) + 1e-150))

		# calculate matrix of ones in y-th positions
		indicatorf = np.zeros(shape=(num_classes, num_images))
		if num_images == 1:
			indicatorf[y] = 1
		else:
			indicatorf[y, np.arange(num_images)] = 1
		# gradient of softmax loss
		dsm = probs - indicatorf

		# calculte gradient (averaged over all images)
		grad = np.dot(dsm, X.T) / num_images

	else:
		print "Incorrect loss function type"
		loss = 0

	# add regularization loss if required
	if reg:
		loss += 0.5 * gamma * np.sum(W**2)
		grad += gamma * W

	return loss, grad
