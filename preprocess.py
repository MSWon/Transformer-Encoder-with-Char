# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:13:33 2018

@author: jbk48
"""

import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
import pickle
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer

class Preprocess():
    
    def __init__(self, char_dim, max_sent_len, max_char_len):
        self.char_dim = char_dim
        self.max_sent_len = max_sent_len
        self.max_char_len = max_char_len
        self.stop = set(stopwords.words('english'))
        self.stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', ""])


    def load_data(self, train_filename, test_filename, max_len):
        print("Making corpus!\nCould take few minutes!") 
        corpus, labels = self.read_data(train_filename)
        self.train_X, self.train_seq_length, self.train_Y = self.clean_text(corpus, labels)
        corpus, labels = self.read_data(test_filename)
        self.test_X, self.test_seq_length, self.test_Y = self.clean_text(corpus, labels)
        print("Tokenize done!")
        return self.train_X, self.train_seq_length, self.train_Y, self.test_X, self.test_seq_length, self.test_Y
    
    def read_data(self, filename):
        
        data = pd.read_csv(filename)      
        labels = data.iloc[:,0]
        corpus = data.iloc[:,2]    
        encoder = LabelBinarizer()
        encoder.fit(labels)
        labels = encoder.transform(labels)
        labels = np.array([np.argmax(x) for x in labels])         
        return corpus, labels
    
    def clean_text(self, corpus, labels):             
        tokens = []
        index_list = []
        seq_len = []
        index = 0
        for sent in corpus:
            text = re.sub('<br />', ' ', sent)
            text = re.sub('[^a-zA-Z]', ' ', sent)
            t = [token for token in tokenize.word_tokenize(text) if not token in self.stop and len(token)>1 and len(token)<=20]

            if(len(t) > self.max_sent_len):
                t = t[0:self.max_sent_len]

            if(len(t) > 10):
                seq_len.append(len(t))
                t = t + ['<pad>'] * (self.max_sent_len - len(t)) ## pad with max_len
                tokens.append(t)
                index_list.append(index)
            index += 1
            
        labels = labels[index_list]
        return tokens, seq_len, labels
    
    def prepare_embedding(self, char_dim):
        self.get_word_embedding() ## Get pretrained word embedding        
        tokens = self.train_X + self.test_X        
        self.get_char_list(tokens)  ## build char dict 
        self.get_char_embedding(char_dim, len(self.char_list)) ## Get char embedding
        return self.word_embedding, self.char_embedding
        
    def prepare_data(self, input_X, input_Y, mode):
        ## Data -> index
        input_X_index = self.convert2index(input_X, "UNK")
        input_X_char, input_X_char_len = self.sent2char(input_X, mode)
        input_X_index = np.array(input_X_index)
        input_Y = np.array(input_Y)
        return input_X_index, input_X_char, input_X_char_len, input_Y

    def get_word_embedding(self, filename = "polyglot-en.pkl"):
        print("Getting polyglot embeddings!")
        words, vector = pd.read_pickle(filename)  ## polyglot-en.pkl
        words = ['<pad>'] + list(words)  ## add PAD ID
        vector = np.append(np.zeros((1,64)),vector,axis=0)
        self.vocabulary = {word:index for index,word in enumerate(words)}
        self.reverse_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        self.index2vec = vector
        self.word_embedding = tf.get_variable(name="word_embedding", shape=vector.shape, initializer=tf.constant_initializer(vector), trainable=True)
    
    def convert2index(self, doc, unk = "UNK"):
        word_index = []
        for sent in doc:
            sub = []
            for word in sent:
                if(word in self.vocabulary):
                    index = self.vocabulary[word]
                    sub.append(index)
                else:
                    if(unk == "UNK"):
                        unk_index = self.vocabulary["<UNK>"]
                        sub.append(unk_index)   
            word_index.append(sub)           
        return word_index

    def get_char_list(self,tokens):
        if os.path.exists("./char_list.csv"):
            char_data = pd.read_csv("./char_list.csv", sep = ",", encoding='CP949')
            char = list(char_data.iloc[:,1])
            print("char_list loaded!")
        else:
            t = []
            for token in tokens:
                t += token
            t = np.array(t)
            s = [list(set(chain.from_iterable(elements))) for elements in t]
            s = np.array(s).flatten()
            char = list(set(chain.from_iterable(s)))
            char = sorted(char)
            char = ["<pad>"] + char
            c = pd.DataFrame(char)
            c.to_csv("./char_list.csv", sep = ",")
            print("char_list saved!")
        
        self.char_list = char
        self.char_dict = {char:index for index, char in enumerate(self.char_list)}


    def sent2char(self, inputs, train = "train"): ## inputs : [batch_size, max_sent_len]
        
        if os.path.exists("./sent2char_{}.pkl".format(train)):
            with open("./sent2char_{}.pkl".format(train), 'rb') as f:
                outputs,char_len = pickle.load(f)
        else:
            char_len, outputs = [], []
            for sent in inputs:
                sub_char_len, sub_outputs = [], []
                for word in sent:
                    if word == "<pad>":
                        sub_char_len.append(0)
                        sub_outputs.append([0]*self.max_char_len)
                    else:
                        if(len(word) > self.max_char_len):
                            word = word[:self.max_char_len]
                        sub_char_len.append(len(word))
                        sub_outputs.append([self.char_dict[char] for char in word])
                outputs.append(pad_sequences(sub_outputs, maxlen = self.max_char_len, padding = "post"))
                char_len.append(sub_char_len)
            
            outputs = np.array(outputs)
            char_len = np.array(char_len)
            results = (outputs,char_len)
            with open("./sent2char_{}.pkl".format(train), 'wb') as f:
                pickle.dump(results , f)
            
        return outputs,char_len
                
    def get_char_embedding(self, embedding_size, vocab_size):
        self.char_embedding = tf.get_variable('char_embedding', [vocab_size, embedding_size])
        self.clear_char_embedding_padding = tf.scatter_update(self.char_embedding, [0], tf.constant(0.0, shape=[1, embedding_size]))
