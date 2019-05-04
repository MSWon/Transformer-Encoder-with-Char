# Transformer-Encoder-with-Char
1. **Transformer Encoder with Char information** for text classification
2. This code was created by referring to the code in [carpedm20](https://github.com/carpedm20/lstm-char-cnn-tensorflow) and [DongjunLee](https://github.com/DongjunLee/transformer-tensorflow)

## 1. Model structure
![alt text](https://github.com/MSWon/Transformer-Encoder-with-Char/blob/master/images/model_structure.png "Model")

1. Input words are represented with **Char-CNN**, **Word2vec** concatenated together(**64 dimensions each**)

2. Normal Transformer Encoder from ([Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)) is used

3. Model is composed of **7 Transformer Encoder layers** with **4 attention heads**

4. **Global Average Pooling** layer with softmax is used at the end, for predicting class 

## 2. Char CNN
![alt text](https://github.com/MSWon/Transformer-Encoder-with-Char/blob/master/images/char_cnn.PNG "Char CNN")

1. Char CNN implemented by [Yoon Kim](https://arxiv.org/pdf/1508.06615.pdf)

## 3. Prerequisite
- [Tensorflow 1.8.0](https://www.tensorflow.org/)
- Python 3.6

## 4. Training
1. Clone git
```
$ git clone https://github.com/MSWon/Transformer-Encoder-with-Char.git
```
2. Unzip **data.zip** and **embedding.zip**
```
$ unzip data.zip
$ unzip embedding.zip
```
3. Training with custom settings (char_mode : (char_cnn, char_lstm, no_char))
```
$ python train.py --batch_size 128 --training_epochs 12 --char_mode char_cnn
```

## 5. Experiments

### 5-1. Datasets

1. The **AG’s news** topic classification dataset is constructed by choosing 4 largest classes from the original news corpus
2. 4 classes are ‘world’, ‘sports’, ‘business’ and ‘science/technology’
3. Each class contains 30,000 training samples and 1,900 testing samples
4. The total number of training samples is **120,000** and **7,600** for test

### 5-2. Test loss graph
![alt text](https://github.com/MSWon/Transformer-Encoder-with-Char/blob/master/images/test_loss_graph.png "loss graph")

### 5-3. Performance table
![alt text](https://github.com/MSWon/Transformer-Encoder-with-Char/blob/master/images/performance_table.png "table")
