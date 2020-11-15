#!/usr/bin/env python

# implement the networks given input and output shape (and sess?)
# simple frame : dense layers
# image frame : CNN layer + dense
# memory frame : GRU / LSTM layer + dense
# image memory frame : CNN layer + GRU / LSTM layer + dense

# essential imports for this Neural Networks frameworks
import tensorflow as tf
from tensorflow.keras import layers
# other necessary imports
from typing import Any, List, Sequence, Tuple
import tqdm
import numpy as np
import collections

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()
# to set the tf random seed for reproducibility
def set_tf_seed(seed=1):
	tf.random.set_seed(seed)

# a simple frame with just two dense layers
class Simple_Frame(tf.keras.Model):
		"""Simple_Frame deep neural network with dense layers
		"""
		def __init__(self, output_size: int, n_num_hidden_units: List[int] = [64, 128, 128, 64]):
			super(Simple_Frame, self).__init__()

			# do the number of requested hidden dense layers with RELU activation function
			self.hidden_layers = []
			for num_hidden_units in n_num_hidden_units:
				self.hidden_layers.append(layers.Dense(num_hidden_units, activation="relu"))
			# and set up the output layer
			self.output_layer = layers.Dense(output_size)

		def call(self, inputs: tf.Tensor) -> tf.Tensor:
			# pass inputs on model and return the output value Tensor
			grinded = inputs
			for hidden_layer in self.hidden_layers:
				grinded = hidden_layer(grinded)
			return self.output_layer(grinded)
				