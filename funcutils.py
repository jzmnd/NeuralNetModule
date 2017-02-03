#! /usr/bin/env python
"""
funcutils.py
General Mathematical Function Utilities

 -- Activation functions and their gradients

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

from __future__ import division
import numpy as np


def tanhf(x):
	return np.tanh(x)


def reLUf(x):
	return x * (x > 0)


def leakyreLUf(x, alpha=0.001):
	return x * (x > 0) + alpha * x * (x < 0)


def tanhg(x):
	return 1.0 - (np.tanh(x))**2


def reLUg(x):
	return 1.0 * (x > 0)


def leakyreLUg(x, alpha=0.001):
	return 1.0 * (x > 0) + alpha * (x <= 0)
