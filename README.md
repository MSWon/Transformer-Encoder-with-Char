# Transformer-Encoder-with-Char
**Transformer Encoder with Char** information for text classification

## 1. Model structure
![alt text](https://github.com/MSWon/Transformer-Encoder-with-Char/blob/master/images/model_structure.png "Model")

1. Input words are represented with Char-CNN, Word2vec concatenated together

2. Normal Transformer Encoder from ([Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)) is used

3. Model is composed of 7 Transformer Encoder layers with 4 attention heads and Global average pooling layer at the end
