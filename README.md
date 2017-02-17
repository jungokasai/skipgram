## A TensorFlow Implementation of Skip-gram with Negative Sampling
----

### Requirements
TensorFlow needs to be installed before running the training script.
TensorFlow 0.11 and 0.12 are supported. We will test it on TensorFlow 1.0 soon.

### Introduction
Skip-gram is a method of word embedding based on a shallow feed-forward neural network, proposed by [Mikolov et al](https://arxiv.org/pdf/1301.3781.pdf) in 2013. The spirit of the algorithm is to characterize a word by the distribution of words around it, which is often called the window of the word. It also aligns with the idea of "bag of words"  in the sense that we do not take into account the order of the words in the window. The network basically learns to predict words around a given word. 

### Preprocessing and Corpus
Our default data set is [English Wikipedia 2017](https://dumps.wikimedia.org/other/incr/wikidatawiki/). [The standard processing](http://mattmahoney.net/dc/textdata.html) for word vectorization has to be performed.

### Negative Sampling
In the task of word prediction, the last softmax layer becomes computationally expensive. In order to speed up this process, several methods have been proposed, such as importance sampling, noise contrasitive estimation (NCE), and negative sampling. TensorFlow provides a off-the-shelf function for NCE, but not negative sampling. For this reason, we hard-code it in the computation graph. Note that the negative samples are sampled from the unigram distribution of words in the corpus following [Mikolov et al](https://arxiv.org/pdf/1301.3781.pdf). The default number of negative sample is set to 5, but you can change it by the -k option. 

### Usage 
Before running the model, you need to have a sub-directory, data/. For instance, in your clone repository, run commands something like:
```bash
mkdir data/
``` 
to make the data available by the relative path. 
Then, run the main code. For example, 

```bash
python word2vec_main.py train -k 10 
```
To get the trained embedding matrix run a command:
```bash
python extract_weights.py <PATH_TO_YOUR_MODEL>
``` 
where \<PATH\_TO\_YOUR\_MODEL\> is, for example, ../word2vec_models/clean_train/word2vec_embeddim300_seed0_dropout1.0_windowsize5_numsamples10_vocabsize400000_lrate0.025_maxepochs5/epoch0.01_loss5.0307. It is lengthy so that you will know the exact configuration of the saved model. 

To check what options are available for the configuration of your model, run
```bash
python word2vec_main.py train -h 
```

### Notes on Saver
For the English Wikipedia 2017 corpus, we save a model every 0.01 epochs since one epoch typically needs 24+ hours. Otherwise, we save it at the end of every epoch.

### Notes on Optimization Algorithms
There are many optimization algorithms for neural networks. We implemented stochastic gradient descent with linear decay, roughly following [Mikolov et al](https://arxiv.org/pdf/1301.3781.pdf).  Unfortunately, [Mikolov et al](https://arxiv.org/pdf/1301.3781.pdf) do not mentions details of their configuration, and therefore, the optimization algorithm could be different. The learning rate of gradient descent decays in a way that at the end of the last training batch in the last training epoch, the rate becomes zero. However, this scheduling, of course, depends upon the batch size, which [Mikolov et al](https://arxiv.org/pdf/1301.3781.pdf) do not provide. And it is also not clear how often they update their learning rate. Our implementation updates the learning rate evenly by an increment -0.001 so that at end of the last training batch of the last epoch, the rate becomes zero.

