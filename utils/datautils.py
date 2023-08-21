#! /usr/bin/env python
"""
datautils.py
Data Utility Functions

-- Functions for extracting CIFAR data and preprocessing
-- Functions for loading NN architecture model and config yaml file
-- Preprocessing function
-- Weight initialization function

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

from __future__ import division

import os

import cPickle as pickle
import numpy as np
import yaml


def load_CIFAR_batch(filename):
    """Load single batch of CIFAR"""
    with open(filename, "r") as f:
        datadict = pickle.load(f)
        y = np.array(datadict["labels"]).astype("int")
        x = datadict["data"].T.astype("float64")
    return x, y


def biastrick(X):
    """Take X matrix and adds 1s to end of each image column"""
    num_images = X.shape[1]
    return np.vstack([X, np.ones(num_images)])


def load_arch(filename, path="architectures-configs"):
    """Loads yaml architecture file and returns model as python dictionary"""
    with open(os.path.join(path, filename), "r") as f:
        modeldict = yaml.load(f)
    nLayers = len(modeldict["layers"])
    return modeldict, nLayers


def load_config(filename, path="architectures-configs"):
    """Loads yaml config file and returns python dictionary"""
    with open(os.path.join(path, filename), "r") as f:
        configdict = yaml.load(f)
    return configdict


def preprocess(x, norm=None):
    """Subtract mean for each column and perform normalization if required"""
    mean = np.mean(x, axis=0)
    x -= mean
    if norm == "std":
        xnorm = np.std(x, axis=0)
    if norm == "uni":
        xnorm = np.max(abs(x), axis=0)
    else:
        xnorm = 1.0
    x /= xnorm
    return x, mean, xnorm


def initializeweights(num_images, num_dims, k=1, a=2.0, b=0.01):
    """Initialize weights for array of size num_dims x k"""
    if k == 1:
        return b * np.random.randn(num_dims).astype("float64") * np.sqrt(a / num_images)
    else:
        return b * np.random.randn(k, num_dims).astype("float64") * np.sqrt(a / num_images)
