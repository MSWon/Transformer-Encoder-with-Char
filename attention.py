# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:53:17 2019

@author: jbk48
"""


import numpy as np
import tensorflow as tf
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D


__all__ = [
    "positional_encoding", "Attention"
]


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    encoded_vec = encoded_vec.reshape([sentence_length, dim])
    encoded_vec = np.append(encoded_vec,np.zeros((1,dim)),axis=0)
    return tf.convert_to_tensor(encoded_vec, dtype=dtype)


class Attention:

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100,
                 dropout=0.2,
                 batch_size=128):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout
        self.batch_size = batch_size

    def multi_head(self, q, k, v, seq_len):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, seq_len)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def classifier_head(self, q, k, v, seq_len):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, seq_len)
        output = self._GlobalAverage_heads(outputs)
        
        return output

    def _GlobalAverage_heads(self, outputs):
        outputs = tf.transpose(outputs, [0, 3, 2, 1]) # [batch_size, dim, max_seq_len, num_heads]
        outputs = GlobalAveragePooling2D()(outputs)
        return outputs

    def _GlobalMax_heads(self, outputs):
        outputs = tf.transpose(outputs, [0, 3, 2, 1]) # [batch_size, dim, max_seq_len, num_heads]
        outputs = GlobalMaxPooling2D()(outputs)
        return outputs

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim): ## dim = num_head * project_dim
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs, seq_len):
        ## qs, ks, vs : [batch_size, num_heads, max_seq_len, dim]
        key_dim_per_head = self.linear_key_dim // self.num_heads
        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5) ## [batch_size, num_heads, max_seq_len, max_seq_len]
        
        if self.masked: ## mask score matrix to max_seq_len
            row_vector = tf.range(0,o2.shape[2],1)  ## [, max_seq_len]
            matrix = tf.cast(tf.expand_dims(seq_len,-1), tf.int32) ## [batch_size, 1]
            
            t = tf.cast(row_vector < matrix, tf.float32) ##  [batch_size, max_seq_len]
            t = tf.expand_dims(t, -1) ##  [batch_size, max_seq_len, 1]
            masks = t * tf.transpose(t, [0,2,1]) ##  [batch_size, max_seq_len, max_seq_len]
            masks = tf.tile(tf.expand_dims(masks, 1), [1, int(o2.shape[1]), 1, 1]) ##  [batch_size, num_heads, max_seq_len, max_seq_len]

            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)
            
        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim]) # [batch_size, max_seq_len, num_heads*dim]

        return transpose_then_concat_last_two_dimenstion(outputs)