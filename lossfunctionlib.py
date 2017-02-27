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
import numpy as np

__author__ = "Jeremy Smith"
__version__ = "1.2"


def loss_function(scores, y, ftype='svm', delta=1.0):
	"""
	Loss function implementation (fully vectorized, also works if number of images is 1)
	-- scores: holds all the input scores as columns (e.g. 10 x 50,000 in CIFAR-10)
	-- y: array of integers specifying correct class (e.g. 50,000-D array)
	-- W: weights (e.g. 10 x 3073)
	-- type: can be either 'svm' or 'softmax'
	Returns
	-- Loss scalar (svm or softmax)
	-- Gradient of loss wrt scores
	"""

	num_images = y.size

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

		# calculte gradient of svm loss wrt scores (averaged over all images)
		grad = indicatorf / num_images

	elif ftype == 'softmax':
		# first shift scores so highest value is 0 and calculate exp
		scores_shift = scores - np.max(scores)
		exp_scores = np.exp(scores_shift)

		# calculate probabilities (allows for 0 sum)
		probs = exp_scores / np.sum(exp_scores, axis=0) + 1e-150

		# compute the cross entropy loss for each image
		if num_images == 1:
			loss_i = -np.log(probs[y])
		else:
			loss_i = -np.log(probs[y, np.arange(num_images)])

		# calculte loss (averaged over all images)
		loss = np.sum(loss_i) / num_images

		# calculate matrix of ones in y-th positions
		indicatorf = np.zeros(scores.shape)
		if num_images == 1:
			indicatorf[y] = 1
		else:
			indicatorf[y, np.arange(num_images)] = 1

		# calculate gradient of softmax loss wrt scores (averaged over all images)
		grad = (probs - indicatorf) / num_images

	else:
		print "Incorrect loss function type"
		loss = 0
		grad = np.zeros(scores.shape)

	return loss, grad
