# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:33:17 2019

@author: jbk48
"""

import tensorflow as tf

from attention import Attention
from layer import FFN


class Encoder:

    def __init__(self,
                 num_layers=6,
                 num_heads=8,
                 linear_key_dim=32*8,
                 linear_value_dim=32*8,
                 model_dim=64,
                 ffn_dim=64,
                 dropout=0.2,
                 n_class=4,
                 batch_size=128):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.n_class = n_class
        self.batch_size = batch_size

    def build(self, encoder_inputs, seq_len):
        o1 = tf.identity(encoder_inputs)

        for i in range(1, self.num_layers+1):
            with tf.variable_scope("layer-{}".format(i)):
                o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                                 k=o1,
                                                                 v=o1,
                                                                 seq_len=seq_len), num=1)
                o3 = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
                o1 = tf.identity(o3)
          
        with tf.variable_scope("GlobalAveragePooling-layer"):
            o3 = self._pooling_layer(q=o1, k=o1, v=o1, seq_len =seq_len)
            
        return o3


    def _pooling_layer(self, q, k, v, seq_len):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout,
                                    batch_size=self.batch_size)
            return attention.classifier_head(q, k, v, seq_len)

    def _self_attention(self, q, k, v, seq_len):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout,
                                    batch_size=self.batch_size)
            return attention.multi_head(q, k, v, seq_len)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope("add-and-norm-{}".format(num)):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_gelu_dense(output)