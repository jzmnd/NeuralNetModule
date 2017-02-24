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

from __future__ import division
import os
import sys
from myfunctions import *
import numpy as np
import random as rnd
from optutils import *
from funcutils import *

__author__ = "Jeremy Smith"
__version__ = "1.1"


def grad_check_sparse(f, x, analytic_grad=0, num_checks=3, step=1e-6, verbose=False):
	"""
	Samples random elements and returns numerical grad in these dimensions
	-- f: single argument function
	-- x: x values for function f
	-- analytic_grad: analytic gradient to compare to numerical gradient
	-- num_checks: number of points to check
	-- step: step size for numerical gradient calculation
	"""

	av_rel_error = 0

	for i in xrange(num_checks):
		ix = tuple([np.random.randint(m) for m in x.shape])
		x0 = x[ix]

		# evaluate at x + step
		x[ix] = x0 + step
		fxp = f(x)
		# evaluate at x - step
		x[ix] = x0 - step
		fxm = f(x)
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


def grad_descent(X, y, W, config, lossf):
	"""
	Performs the gradient descent and weights update
	-- X: holds all the data as columns (e.g. 3073 x 50,000 in CIFAR-10)
	-- y: array of integers specifying correct class (e.g. 50,000-D array)
	-- W: weights (e.g. 10 x 3073)
	-- config: dictionary containing all the configuration options
	-- lossf: loss function (function that takes X, y, W and returns loss, dW)
	-- Has options for 'full', 'minibatch' or 'stochastic'
	-- Has options for weight update function from optutils
	-- Has options for 'svm' or 'softmax' loss functions
	"""

	n = y.size
	count = 1
	loss = np.inf
	gradchecks = []
	losses = []
	accuracies = []

	update_function = methods[config['update_type']]

	while (loss > config['maxloss']) and (count <= config['maxsteps']):
		if config['btype'] == 'minibatch' or config['btype'] == 'stochastic':
			if config['btype'] == 'stochastic':
				mask = rnd.sample(xrange(n), 1)
			else:
				mask = rnd.sample(xrange(n), config['batch_size'])
			Xbatch = X[:, mask]
			ybatch = y[mask]
		else:
			Xbatch = X
			ybatch = y

		# calculate loss and gradient on loss wrt weights
		loss, dW = lossf(Xbatch, ybatch, W)

		# define a function that takes a single arguement i.e. the weights and returns a single value i.e. the loss
		f = lambda w: lossf(Xbatch, ybatch, w)[0]
		# check gradient numerically for a few points
		gradcheck = grad_check_sparse(f, W, analytic_grad=dW, num_checks=config['num_checks'], step=config['gradcheckstep'])

		# update weights using update_function, also calculates weights:updates ratio and updates config
		W, config, w_u_ratio = update_function(W, dW, config)

		# calculate accuracy on fly
		scores = score_function(X, W)
		predicted_classes = np.argmax(scores, axis=0)
		accuracy = np.mean(predicted_classes == y) * 100

		if config['verbose']:
			if (count % 10 == 0) or (count == 1):
				print "step: {:4d}   loss: {:.5e}   w/u ratio: {:.3e}   gradcheck: {:.3e}   accuracy: {:.2f} %".format(count, loss, w_u_ratio, gradcheck, accuracy)

		gradchecks.append(gradcheck)
		accuracies.append(accuracy)
		losses.append(loss)
		count += 1

	return np.array(losses), W, dW, np.array(gradcheck), np.array(accuracies), count
