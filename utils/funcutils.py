#! /usr/bin/env python
"""
funcutils.py
General Mathematical Function Utilities

-- Activation functions and their gradients
-- Score functions

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

from __future__ import division
import numpy as np


def tanhf(x):
	"""Tanh activation function"""
	return np.tanh(x)


def reLUf(x):
	"""ReLU activation function"""
	return x * (x > 0)


def leakyreLUf(x, alpha=0.001):
	"""Leaky ReLU activation function"""
	return x * (x > 0) + alpha * x * (x < 0)


def tanhg(x):
	"""Tanh gradient function"""
	return 1.0 - (np.tanh(x))**2


def reLUg(x):
	"""ReLU gradient function"""
	return 1.0 * (x > 0)


def leakyreLUg(x, alpha=0.001):
	"""Leaky ReLU gradient function"""
	return 1.0 * (x > 0) + alpha * (x <= 0)


def score_function(x, W):
	"""Simple linear score function"""
	return np.dot(W, x)


def score_function_bias(x, W, b):
	"""Simple linear score function with explicit bias"""
	return np.dot(W, x) + b


functions = {'reLU': reLUf, 'tanh': tanhf, 'leakyreLU': leakyreLUf}
gradientfunctions = {'reLU': reLUg, 'tanh': tanhg, 'leakyreLU': leakyreLUg}
