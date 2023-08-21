#! /usr/bin/env python
"""
nnsolver.py
Solver class for performing gradient descent training on a model

Based on CS231n course by Andrej Karpathy
Created by Jeremy Smith on 2016-04-07
University of California, Berkeley
j-smith@berkeley.edu
"""

import os
import logging

from graddescentlib import grad_descent
from utils.datautils import load_config
from utils.optutils import methods

__author__ = "Jeremy Smith"
__version__ = "2.0"


class Solver:
    """Class containing solver for training of NN or linear classifier models using gradient descent"""

    def __init__(self, model, data_training, data_val, k=1, config="default.config.yml"):
        self.model = model
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.X_train = data_training[0]
        self.y_train = data_training[1]
        self.X_val = data_val[0]
        self.y_val = data_val[1]
        self.k = k
        self.Xinputdims = self.X_train.shape[0]

        self.num_images_train = self.y_train.size
        self.num_images_val = self.y_val.size
        self.count = 1

        self.gradchecks = []
        self.losses = []
        self.accuracies_train = []
        self.accuracies_val = []

        self.config = load_config(config)

        self.filenametag = "_{:s}_{:.0e}_{:.4f}_{:s}_{:s}_{:s}".format(
            self.model.name,
            self.config["learning_rate"],
            self.config["mu"],
            self.config["btype"],
            self.config["ftype"],
            self.config["update_type"],
        )

        self.update_function = methods[self.config["update_type"]]

    def train(self):
        """Perform training on model"""
        logging.info("TRAINING ON DATA")
        logging.info("  model name         : %s", self.model.name)
        logging.info("  model type         : %s", self.model.__class__.__name__)
        logging.info("  num of points      : %s", self.num_images_train)
        logging.info("  num of dimensions  : %s", self.Xinputdims)
        logging.info("  num of classes     : %s", self.k)
        logging.info("  solver update type : %s", self.config["update_type"], self.config["btype"])
        logging.info("  init learning rate : %s", self.config["learning_rate"])
        logging.info("  loss function      : %s", self.config["ftype"])

        self.model.init_weights(
            self.config["aini"], self.config["bini"], self.num_images_train, self.Xinputdims, self.k
        )

        logging.info("GRADIENT DESCENT")
        results = grad_descent(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.model.weights,
            self.config,
            self.model.compute_loss,
            self.model.forward,
        )
        (
            self.losses,
            self.model.weights,
            self.dW,
            self.gradchecks,
            self.accuracies_train,
            self.accuracies_val,
            self.count,
        ) = results
        return
