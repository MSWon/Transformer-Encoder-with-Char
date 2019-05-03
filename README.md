# Transformer-Encoder-with-Char
**Transformer Encoder with Char** information for text classification
This code was created by referring to the code in [carpedm20](https://github.com/carpedm20/lstm-char-cnn-tensorflow) and [DongjunLee](https://github.com/DongjunLee/transformer-tensorflow)

## 1. Model structure
![alt text](https://github.com/MSWon/Transformer-Encoder-with-Char/blob/master/images/model_structure.png "Model")

1. Input words are represented with Char-CNN, Word2vec concatenated together(64 dimensions each)

2. Normal Transformer Encoder from ([Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)) is used

3. Model is composed of 7 Transformer Encoder layers with 4 attention heads

4. Global Average Pooling layer with softmax is used at the end for predicting class 

## 2. Prerequisite
- [Tensorflow 1.8.0](https://www.tensorflow.org/)
- Python 3.6
