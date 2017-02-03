#! /usr/bin/env python
"""
graddescentlib.py
Gradient Descent Library Module

 -- Contians functions for gradient descent and numerical gradient calculations

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

import os
import sys
from myfunctions import *
import numpy as np
import random as rnd
from lossfunctionlib import *
from optutils import *
from __future__ import division

__author__ = "Jeremy Smith"
__version__ = "1.0"


def eval_num_grad(f, x, step=1e-6):
	"""Center difference numerical gradient"""

	grad = np.zeros(x.shape)

	# iterate over all indexes in x
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		ix = it.multi_index
		x0 = x[ix]

		# evaluate at x + step
		x[ix] = x0 + step
		fxp = f(x)
		# evaluate at x - step
		x[ix] = x0 - step
		fxm = f(x)
		# return to original value
		x[ix] = x0
		# compute the partial derivative
		grad[ix] = 0.5 * (fxp - fxm) / step

		it.iternext()

	return grad


def grad_check_sparse(f, x, analytic_grad, num_checks=3, step=1e-6, verbose=False):
	"""Samples random elements and returns numerical grad in these dimensions"""

	av_rel_error = 0

	for i in xrange(num_checks):
		ix = tuple([np.random.randint(m) for m in x.shape])
		x0 = x[ix]

		# evaluate at x + step
		x[ix] = x0 + step
		fxp = f(x)[0]
		# evaluate at x - step
		x[ix] = x0 - step
		fxm = f(x)[0]
		# return to original value
		x[ix] = x0
		# compute the numerical derivative
		grad_numerical = 0.5 * (fxp - fxm) / step
		# get the analytical gradient
		grad_analytic = analytic_grad[ix]

		rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
		av_rel_error += rel_error

		if verbose:
			print "    numerical grad: {:.5f}, analytic grad: {:.5f}, relative error: {:.1e}".format(grad_numerical, grad_analytic, rel_error)

	return av_rel_error / num_checks


def grad_descent(X, y, W, config):
	"""
	Performs the gradient descent
	-- X: holds all the data as columns (e.g. 3073 x 50,000 in CIFAR-10)
	-- y: array of integers specifying correct class (e.g. 50,000-D array)
	-- W: weights (e.g. 10 x 3073)
	-- config: dictionary containing all the configuration options
	-- Has options for full, minibatch or stochastic
	-- Has options for weight update function from optutils
	-- Has options for 'svm' or 'softmax' loss functions
	"""

	num_images = y.size
	weights = W

	count = 1
	loss = np.inf
	gradcheck = []
	losses = []
	accuracies = []

	update_function = methods[config['update_type']]

	while (loss > maxloss) and (count <= maxsteps):
		if config['btype'] == 'minibatch' or config['btype'] == 'stochastic':
			if config['btype'] == 'stochastic':
				mask = rnd.sample(xrange(num_images), 1)
			else:
				mask = rnd.sample(xrange(num_images), config['batch_size'])
			Xbatch = X[:, mask]
			ybatch = y[mask]
		else:
			Xbatch = X
			ybatch = y

		# define a loss function that takes a single arguement i.e. the weights
		lf = lambda w: loss_function(Xbatch, ybatch, w, ftype=config['ftype'], reg=config['reg'], gamma=config['gamma'])

		# calculate loss and gradient for weights
		loss, dW = lf(weights)

		# check gradient numerically for a few points
		gradcheck.append(grad_check_sparse(lf, weights, dW, num_checks=config['num_checks'], step=config['gradcheckstep']))

		# update weights using update_function also calculates weights:updates ratio
		weights, config, w_u_ratio = update_function(weights, dW, config)

		# calculate accuracy on fly
		scores = score_function(X, weights)
		predicted_classes = np.argmax(scores, axis=0)
		accuracy = np.mean(predicted_classes == y) * 100

		if config['verbose']:
			if (count % 10 == 0) or (count == 1):
				print "step: {:4d}   loss: {:.5e}   w/u ratio: {:.3e}   accuracy: {:.2f} %".format(count, loss, w_u_ratio, accuracy)

		accuracies.append(accuracy)
		losses.append(loss)
		count += 1

	return np.array(losses), weights, dW, np.array(gradcheck), np.array(accuracies), count
