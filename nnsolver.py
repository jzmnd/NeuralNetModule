#! /usr/bin/env python
"""
nnsolver.py
Solver class for performing gradient descent training on a NN model

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


class Solver():
	"""Class containing solver for training of NN using gradient descent"""
	def __init__(self, model, data_training, data_val):
		self.model = model
		self.X_train = data_training[0]
		self.y_train = data_training[1]
		self.X_val = data_val[0]
		self.y_val = data_val[1]

		self.num_images_train = y_train.size
		self.num_images_val = y_val.size
		self.count = 1

		self.gradchecks = []
		self.losses = []
		self.accuracies_train = []
		self.accuracies_val = []

	def train(self):

