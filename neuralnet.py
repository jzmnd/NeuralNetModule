#! /usr/bin/env python
"""
neuralnet.py
Neural Network Classes

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
import yaml

__author__ = "Jeremy Smith"
__version__ = "1.0"


class FeedForwardNN():
	"""Class for descibing basic feed forward neural network"""
	def __init__(self, weights=None, architecture="default.arch.yml"):
		self.weights = weights
		self.architecture = load_arch(architecture)
		self.name = self.architecture['name']
		self.layers = sorted(self.architecture['layers'], key=lambda l: l['id'])

	def forward(self):






class RecursiveNN():
	"""Class for descibing recursive neural network"""
	def __init__(self, weights, bias):


class ConvNN():
	"""Class for descibing convolutional neural network"""
	def __init__(self, weights, bias):
