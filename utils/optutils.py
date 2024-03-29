#! /usr/bin/env python
"""
optutils.py
Functions for Optimising Gradient Descent

-- Calculates new weights from original weights and dw
-- Updates config dictionary with optimising parameters
-- Calculates ratio of norm(weight):norm(update)

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

from __future__ import division

import numpy as np


def gflatten(x):
    """Generalized flatten function for ndarrays"""
    if x.dtype == "float64":
        return x.ravel()
    if x.dtype == "object":
        return np.hstack([xi.ravel() for xi in x])


def g_zeros_like(x):
    """Generalized zeros_like for ndarrays"""
    if x.dtype == "float64":
        return np.zeros_like(x)
    if x.dtype == "object":
        return np.array([np.zeros_like(xi) for xi in x])


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent

    config format:
    -- learning_rate: Scalar learning rate
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    weights_scale = np.linalg.norm(gflatten(w))

    update = config["learning_rate"] * dw
    update_scale = np.linalg.norm(gflatten(update))

    w -= update

    return w, config, abs(update_scale / weights_scale)


def momentum(w, dw, config=None):
    """
    Performs momentum update

    config format:
    -- learning_rate: Scalar learning rate
    -- mu: Mass parameter
    -- v: Velocity
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("mu", 0.9)
    config.setdefault("v", 0)

    weights_scale = np.linalg.norm(gflatten(w))

    v = config["v"]
    v = config["mu"] * v - config["learning_rate"] * dw

    update_scale = np.linalg.norm(gflatten(v))

    w += v

    config["v"] = v

    return w, config, abs(update_scale / weights_scale)


def nag(w, dw, config=None):
    """
    Performs Nesterov momentum update

    config format:
    -- learning_rate: Scalar learning rate
    -- mu: Mass parameter
    -- v: Velocity
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("mu", 0.9)
    config.setdefault("v", 0)

    weights_scale = np.linalg.norm(gflatten(w))

    v0 = config["v"]
    mu = config["mu"]
    v1 = mu * v0 - config["learning_rate"] * dw

    update = -mu * v0 + (1 + mu) * v1
    update_scale = np.linalg.norm(gflatten(update))

    w += update

    config["v"] = v1

    return w, config, abs(update_scale / weights_scale)


def adagrad(w, dw, config=None):
    """
    Performs an AdaGrad update

    config format:
    -- learning_rate: Scalar learning rate
    -- c: Cache of sum of square gradient
    -- eps: Small scalar used for smoothing to avoid dividing by zero
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("c", g_zeros_like(dw))
    config.setdefault("eps", 1e-8)

    weights_scale = np.linalg.norm(gflatten(w))

    c = config["c"] + dw**2
    alpha = config["learning_rate"] / (c**0.5 + config["eps"])

    update = alpha * dw
    update_scale = np.linalg.norm(gflatten(update))

    w -= update

    config["c"] = c

    return w, config, abs(update_scale / weights_scale)


def adam(w, dw, config=None):
    """
    Performs an Adam update

    config format:
    -- learning_rate: Scalar learning rate
    -- beta1: Decay rate for moving average of first moment of gradient
    -- beta2: Decay rate for moving average of second moment of gradient
    -- eps: Small scalar used for smoothing to avoid dividing by zero
    -- m: Moving average of gradient
    -- v: Moving average of squared gradient
    -- t: Iteration number
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("eps", 1e-8)
    config.setdefault("m", g_zeros_like(w))
    config.setdefault("v", g_zeros_like(w))
    config.setdefault("t", 0)

    weights_scale = np.linalg.norm(gflatten(w))

    beta1, beta2, eps = config["beta1"], config["beta2"], config["eps"]
    t, m, v = config["t"], config["m"], config["v"]
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw**2)
    t += 1

    alpha = config["learning_rate"] * np.sqrt(1 - beta2**t) / (1 - beta1**t)

    update = -alpha * (m / (v**0.5 + eps))
    update_scale = np.linalg.norm(gflatten(update))

    w += update

    config["t"] = t
    config["m"] = m
    config["v"] = v

    return w, config, abs(update_scale / weights_scale)


methods = {"sgd": sgd, "momentum": momentum, "nag": nag, "adagrad": adagrad, "adam": adam}
