# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:55:38 2019

@author: jbk48
"""

import numpy as np
import tensorflow as tf
import transformer
import os
import datetime
import preprocess
import pandas as pd
from attention import positional_encoding

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Model:

    def __init__(self, word_dim, char_dim, max_sent_len, max_char_len, learning_rate, num_train_steps):
        
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.max_sent_len = max_sent_len
        self.max_char_len = max_char_len
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps

        ## Preprocess data
        self.prepro = preprocess.Preprocess(self.char_dim, self.max_sent_len, self.max_char_len)
        self.train_X, self.train_seq_length, self.train_Y, self.test_X, self.test_seq_length, self.test_Y = self.prepro.load_data("./data/train.csv", "./data/test.csv", self.max_sent_len)
        self.word_embedding, self.char_embedding = self.prepro.prepare_embedding(self.char_dim)
        self.train_X, self.train_X_char, self.train_X_char_len, self.train_Y = self.prepro.prepare_data(self.train_X, self.train_Y, "train")
        self.test_X, self.test_X_char, self.test_X_char_len, self.test_Y = self.prepro.prepare_data(self.test_X, self.test_Y, "test")
        
        ## Placeholders
        self.word_input = tf.placeholder(tf.int32, shape = [None, max_sent_len], name = 'word')
        self.char_input = tf.placeholder(tf.int32, shape = [None, max_sent_len, max_char_len], name = 'char')
        self.label = tf.placeholder(tf.int32, shape = [None], name = 'label')
        self.seq_len = tf.placeholder(tf.int32, shape = [None])
        self.char_len = tf.placeholder(tf.int32, [None, max_sent_len])
        self.dropout = tf.placeholder(tf.float32, shape = ())
        
    def train(self, batch_size, training_epochs, char_mode):
        
        self.batch_size = batch_size
        loss, optimizer, logits = self.build_model(self.word_input, self.char_input, self.label, self.seq_len, 
                                                   self.char_len, self.num_train_steps, char_mode)
        accuracy = self.get_accuracy(logits, self.label)        

        ## Training
        init = tf.global_variables_initializer()
        
        num_train_batch = int(len(self.train_X) / self.batch_size)
        num_test_batch = int(len(self.test_X) / self.batch_size)
        print("Start training!")
        
        modelpath = "./transformer_ag_news_{}/".format(char_mode)
        modelName = "transformer_ag_news_{}.ckpt".format(char_mode)
        saver = tf.train.Saver()
        
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []
        
        with tf.Session(config = config) as sess:
        
            start_time = datetime.datetime.now()
            sess.run(init)
            if(not os.path.exists(modelpath)):
                os.mkdir(modelpath)
            ckpt = tf.train.get_checkpoint_state(modelpath)
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                self.load_char_embedding(modelpath + "char_embedding_{}.npy".format(char_mode))
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")
                
            for epoch in range(training_epochs):
        
                train_acc, train_loss = 0., 0.       
                self.train_X, self.train_X_char, self.train_X_char_len, self.train_Y = self.shuffle(self.train_X, 
                                                                                                    self.train_X_char, 
                                                                                                    self.train_X_char_len, 
                                                                                                    self.train_Y)
                for step in range(num_train_batch):
                    if(step == 0):
                        mode = "init"
                    else:
                        mode = None
                    train_batch, train_batch_char, train_batch_char_len, train_batch_Y, train_batch_seq_len = get_batch(self.train_X, 
                                                                                                                        self.train_X_char, 
                                                                                                                        self.train_X_char_len, 
                                                                                                                        self.train_Y, 
                                                                                                                        self.train_seq_length,
                                                                                                                        self.batch_size,
                                                                                                                        mode)           
                    feed_dict_train = {self.word_input: train_batch, self.char_input : train_batch_char, self.label: train_batch_Y,
                                       self.seq_len: train_batch_seq_len, self.char_len: train_batch_char_len, self.dropout : 0.2}
                    
                    char_embedding_matrix = sess.run(self.prepro.clear_char_embedding_padding, feed_dict = feed_dict_train) ## clear 0 index to 0 vector
                    _, train_batch_loss = sess.run([optimizer,loss], feed_dict = feed_dict_train)
                              
                    train_loss += train_batch_loss / num_train_batch          
                    train_batch_acc = sess.run(accuracy , feed_dict = feed_dict_train)
                    train_acc += train_batch_acc / num_train_batch
                    print("epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f}".format(epoch+1, step+1, train_batch_loss, train_batch_acc))
                
                test_acc, test_loss = 0. , 0.
                print("Now for test data\nCould take few minutes")
                for step in range(num_test_batch):
                    if(step == 0):
                        mode = "init"
                    else:
                        mode = None
                    test_batch, test_batch_char, test_batch_char_len, test_batch_Y, test_batch_seq_len = get_batch(self.test_X, 
                                                                                                                   self.test_X_char, 
                                                                                                                   self.test_X_char_len, 
                                                                                                                   self.test_Y, 
                                                                                                                   self.test_seq_length,
                                                                                                                   self.batch_size,
                                                                                                                   mode)
                    feed_dict_test = {self.word_input: test_batch, self.char_input: test_batch_char, self.label: test_batch_Y, 
                                      self.seq_len: test_batch_seq_len, self.char_len: test_batch_char_len, self.dropout : 0.0}
                    # Compute average loss
                    test_batch_loss = sess.run(loss, feed_dict = feed_dict_test)
                    test_loss += test_batch_loss / num_test_batch
                    
                    test_batch_acc = sess.run(accuracy , feed_dict = feed_dict_test)
                    test_acc += test_batch_acc / num_test_batch
                    
                print("<Train> Loss = {:.6f} Accuracy = {:.6f}".format(train_loss, train_acc))
                print("<Test> Loss = {:.6f} Accuracy = {:.6f}".format(test_loss, test_acc))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                np.save(modelpath + "char_embedding_{}.npy".format(char_mode), char_embedding_matrix)
            
            train_loss = pd.DataFrame({"train_loss":train_loss_list})
            train_acc = pd.DataFrame({"train_acc":train_acc_list})
            test_loss = pd.DataFrame({"test_loss":test_loss_list})
            test_acc = pd.DataFrame({"test_acc":test_acc_list})
            df = pd.concat([train_loss,train_acc,test_loss,test_acc], axis = 1)
            df.to_csv("./results_{}.csv".format(char_mode), sep =",", index=False)
            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))
            save_path = saver.save(sess, modelpath + modelName)
            print ('save_path',save_path)

    def char_lstm(self, inputs, char_len, lstm_units, dropout, last=True, scope="char_lstm"): 
        ## inputs : [batch_size, max_sent_len, max_char_len, dim]
        def _build_single_cell(lstm_units, keep_prob):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_units)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0-keep_prob, output_keep_prob=1.0-keep_prob)    
                return cell
        char_len = tf.reshape(char_len, [-1])
        max_sent_len = int(inputs.shape[1])
        max_char_len = int(inputs.shape[2])
        embedding_size = int(inputs.shape[3])
        inputs = tf.reshape(inputs,[-1,max_char_len,embedding_size]) ## [batch_size*max_sent_len, max_char_len, dim]

        with tf.variable_scope("shared_" + scope):
            lstm_cell = _build_single_cell(lstm_units, dropout)

        with tf.variable_scope("birnn-lstm_" + scope):
            _output = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, dtype=tf.float32,
                                                        inputs = inputs, sequence_length = char_len, scope="rnn_" + scope)
            if last:
                _, ((_, output_fw), (_, output_bw)) = _output
                outputs = tf.concat([output_fw, output_bw], axis=1)
                outputs = tf.reshape(outputs, shape=[-1, max_sent_len, 2 * lstm_units])
            else:
                (output_fw, output_bw), _ = _output
                outputs = tf.concat([output_fw, output_bw], axis=2)
                outputs = tf.reshape(outputs, shape=[-1, 2 * lstm_units])
            
            outputs = tf.layers.dense(outputs, self.word_dim)
        return outputs

    def char_cnn(self, input_, kernels, kernel_features, scope='char_cnn'):
        '''
        :input:           input float tensor of shape  [batch_size, max_sent_len, max_word_len, char_embed_size]
        :kernel_features: array of kernel feature sizes (parallel to kernels)
        '''
        assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
        
        max_sent_len = input_.get_shape()[1]
        max_word_len = input_.get_shape()[2]
        char_embed_size = input_.get_shape()[3]
        
        input_ = tf.reshape(input_, [-1, max_word_len, char_embed_size])
    
        input_ = tf.expand_dims(input_, 1) # input_: [batch_size*max_sent_len, 1, max_word_len, char_embed_size]
        
        layers = []
        with tf.variable_scope(scope):
            for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
                reduced_length = max_word_len - kernel_size + 1
    
                # [batch_size*max_sent_len, 1, reduced_length, kernel_feature_size]
                conv = self.conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
    
                # [batch_size*max_sent_len, 1, 1, kernel_feature_size]
                pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
    
                layers.append(tf.squeeze(pool, [1, 2]))
    
            if len(kernels) > 1:
                output = tf.concat(layers, 1) # [batch_size*max_sent_len, sum(kernel_features)]
            else:
                output = layers[0]
            
            # [batch_size, max_sent_len, sum(kernel_features)]
            output = self.highway(output, output.get_shape()[-1], num_layers = 1)
            output = tf.reshape(output, (-1, max_sent_len, sum(kernel_features))) 
            output = tf.layers.dense(output, self.word_dim, activation = None) ## projection layer
            
        return output
    
    def conv2d(self, input_, output_dim, k_h, k_w, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
            b = tf.get_variable('b', [output_dim])
    
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b
    
    
    def highway(self, input_, size, num_layers=1, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """   
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = tf.nn.relu(tf.layers.dense(input_, size, name='highway_lin_%d' % idx))
    
                t = tf.sigmoid(tf.layers.dense(input_, size, name='highway_gate_%d' % idx))
    
                output = t * g + (1. - t) * input_
                input_ = output
    
        return output
    
    def build_parameter(self,num_layers, num_heads, linear_key_dim, linear_value_dim, model_dim, ffn_dim, n_class):
        
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.linear_key_dim=linear_key_dim
        self.linear_value_dim=linear_value_dim
        self.model_dim=model_dim
        self.ffn_dim=ffn_dim
        self.n_class=n_class
        
    def build_model(self, word_inputs, char_inputs, labels, seq_len, char_len, num_train_steps, char_mode):
        print("Building model!")
        if(char_mode == "no_char"):
            self.model_dim /= 2
        
        # Implements linear decay of the learning rate.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
                  self.learning_rate,
                  global_step,
                  num_train_steps,
                  end_learning_rate=0.0,
                  power=1.0,
                  cycle=False)    
        
        encoder = transformer.Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.model_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              n_class=self.n_class,
                              batch_size=self.batch_size)
        encoder_emb = self.build_embed(word_inputs, char_inputs, char_len, char_mode)
        encoder_outputs = encoder.build(encoder_emb, seq_len)
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = encoder_outputs , labels = labels)) # Softmax loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step) # Adam Optimizer
        
        return loss, optimizer, encoder_outputs
    
    def build_embed(self, word_inputs, char_inputs, char_len, char_mode):
        
        # Positional Encoding
        with tf.variable_scope("positional-encoding"):
            positional_encoded = positional_encoding(self.word_dim,
                                                     self.max_sent_len)
        
        
        position_inputs = tf.tile(tf.range(0, self.max_sent_len), [self.batch_size])
        position_inputs = tf.reshape(position_inputs, [self.batch_size, self.max_sent_len]) # batch_size x [0, 1, 2, ..., n]     
        encoded_inputs = tf.add(tf.nn.embedding_lookup(self.word_embedding, word_inputs),
                         tf.nn.embedding_lookup(positional_encoded, position_inputs))
               
        
        if(char_mode == "char_cnn"):
            char_inputs = tf.nn.embedding_lookup(self.char_embedding, char_inputs)
            kernels = [ 1,   2,   3,   4,   5,   6]
            kernel_features = [25, 50, 75, 100, 125, 150]
            char_inputs = self.char_cnn(char_inputs, kernels, kernel_features, scope='char_cnn')
            final_outputs = tf.concat([encoded_inputs,char_inputs], axis=2)
        elif(char_mode == "char_lstm"):
            char_inputs = tf.nn.embedding_lookup(self.char_embedding, char_inputs)
            char_inputs = self.char_lstm(char_inputs, char_len, self.word_dim, self.dropout, last=True, scope="char_lstm")      
            final_outputs = tf.concat([encoded_inputs,char_inputs], axis=2)
        elif(char_mode == "no_char"):
            final_outputs = encoded_inputs
                    
        return final_outputs
    
    def get_accuracy(self, logits, label):
        pred = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct_pred = tf.equal(pred, label)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy
    
    def shuffle(self, train_X, train_X_char, train_X_char_len, train_Y):
        mask = np.random.permutation(len(train_X))
        train_X = train_X[mask]
        train_X_char = train_X_char[mask]
        train_X_char_len = train_X_char_len[mask]
        train_Y = train_Y[mask]
        return train_X, train_X_char, train_X_char_len, train_Y
    
    def load_char_embedding(self, filename):
        print("Char embedding loaded!")
        self.char_embedding = np.load(filename)


step = 0

def get_batch(train_X, train_X_char, train_X_char_len, train_Y, seq_length, batch_size, mode = None):
    global step
    if(mode =="init"):
        step = 0
    train_batch_X = train_X[step*batch_size :]
    train_batch_X_char = train_X_char[step*batch_size :]
    train_batch_X_char_len = train_X_char_len[step*batch_size :]
    train_batch_Y = train_Y[step*batch_size :]
    train_batch_X_seq_len = seq_length[step*batch_size :]
 
    train_batch_X = train_X[step*batch_size : (step+1)*batch_size]
    train_batch_X_char = train_X_char[step*batch_size : (step+1)*batch_size]
    train_batch_X_char_len = train_X_char_len[step*batch_size : (step+1)*batch_size]
    train_batch_Y = train_Y[step*batch_size : (step+1)*batch_size]
    train_batch_X_seq_len = seq_length[step*batch_size : (step+1)*batch_size]
    step += 1
    return train_batch_X, train_batch_X_char, train_batch_X_char_len, train_batch_Y, train_batch_X_seq_len
