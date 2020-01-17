# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:37:55 2019

@author: jbk48
"""

import tensorflow as tf
import numpy as np

class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias

  
def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf



class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self,
                 w1_dim=200,
                 w2_dim=100,
                 dropout=0.1):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout

    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output =tf.layers.dense(output, self.w2_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def dense_gelu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.w1_dim, activation=gelu)
        output =tf.layers.dense(output, self.w2_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def conv_relu_conv(self):
        raise NotImplementedError("i will implement it!")


