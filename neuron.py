#! /usr/bin/env python
"""
neuron.py
Neuron Class

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

import os
import sys
from myfunctions import *
import numpy as np
from linearclassifierlib import *
from graddescentlib import *
from datautils import *
from funcutils import *

__author__ = "Jeremy Smith"
__version__ = "1.0"


class Neuron():
	"""Class for descibing single artificial neuron"""
	def __init__(self, weights, bias, activation_function=reLUf):

	def forward(x):
		cellsum = np.dot(x, self.weights) + self.bias
		output = self.activation_function(cellsum)
		return output

	def backprop(x):

		return
