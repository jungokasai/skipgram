import numpy as np
import os
import sys
from preprocessing import Tokenizer
import collections 
import pickle

np.random.seed(1234)

class Dataset(object):
    def __init__(self, opts):
        f_train = open(opts.data_dir, 'r')
        tokenizer = Tokenizer(opts)
        tokenizer.fit_on_texts(f_train)
        f_train.close()
        self.count = tokenizer.count
        reverse_dictionary = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
        print('Most common words (+UNK)', self.count[:5])
        self.unigrams = [x[1] for x in self.count]
        self.dictionary = tokenizer.word_index
        with open('../word2vec_models/{0}/{1}dictionary.pkl'.format(os.path.basename(opts.data_dir), opts.vocab_size), 'wb') as fhand:
            pickle.dump(self.dictionary, fhand)
        self.reverse_dictionary = reverse_dictionary
        self.num_words = sum([count for _, count in tokenizer.word_counts.items()]) 
        num_centers = self.num_words - 2*opts.window_size
        num_examples = num_centers*opts.window_size*2
        self.num_examples = num_examples//opts.batch_size*opts.batch_size
        self.num_batches = self.num_examples/opts.batch_size
        
        self._epoch_completed = 0 
        self._example_index = 0
        self._batch_index = 0
        self._window_index = 0
        self._relative_window_index = -opts.window_size
        self.center = opts.window_size
        self._index_in_epoch = 0
        self.tokenizer = tokenizer
        #self.window = self.tokenizer.next_window()
        self.opts = opts
        ## starting data partition
        if opts.data_dir == '../data/clean_train':
            self.num_parts = 100 # the entire enwiki is too huge
        else:
            self.num_parts = 4
        self.num_per_partition = self.num_words//self.num_parts 
        self.num_remainders = self.num_words - self.num_per_partition*(self.num_parts-1)
        self.data_partition = self.tokenizer.texts_to_sequences_partition(self.num_per_partition)
        self._partition_index = 0

    def next_batch_less_mem(self):
        self.X_train_batch = np.ndarray(shape=(self.opts.batch_size), dtype=np.int32) 
        self.y_train_batch = np.ndarray(shape=(self.opts.batch_size), dtype=np.int32) 
        for j in xrange(self.opts.batch_size):
            if self._window_index == self.opts.window_size: #skip center
                self._window_index += 1
            if self._window_index >= self.opts.window_size*2+1: 
                self._window_index = 0
                if self._batch_index >= self.num_batches:
                    self._batch_index = 0
                    self._epoch_completed =+1
                    self.tokenizer.reset_window()
                    self._partition_index = 0
                    return False
                self.window = self.tokenizer.next_window()

            self.X_train_batch[j] = self.window[self.opts.window_size]
            self.y_train_batch[j] = self.window[self._window_index]
            self._window_index+=1
        self._batch_index += 1
        return True

    def next_batch(self):
        self.X_train_batch = np.ndarray(shape=(self.opts.batch_size), dtype=np.int32) 
        self.y_train_batch = np.ndarray(shape=(self.opts.batch_size), dtype=np.int32) 
        for j in xrange(self.opts.batch_size):
            if self._relative_window_index == 0:
                self._relative_window_index = 1  # avoid the center
            if self._relative_window_index > self.opts.window_size: # find a new window
                self._relative_window_index = -self.opts.window_size
                self.center += 1
                if self.center >= (len(self.data_partition)-self.opts.window_size):
                    self.center = self.opts.window_size
                    self._partition_index += 1
                    if self._partition_index == self.num_parts-1:
                        self.data_partition = self.tokenizer.texts_to_sequences_partition(self.num_remainders)
                    elif self._partition_index == self.num_parts:
                        self._batch_index = 0
                        self._epoch_completed =+1
                        self._partition_index = 0
                        self.tokenizer.reset_window()
                        self.data_partition = self.tokenizer.texts_to_sequences_partition(self.num_per_partition)
                        return False
                    else: 
                        self.data_partition = self.tokenizer.texts_to_sequences_partition(self.num_per_partition)
            self.X_train_batch[j] = self.data_partition[self.center]
            self.y_train_batch[j] = self.data_partition[self.center+self._relative_window_index]
            self._relative_window_index += 1
        self._batch_index+=1
        #print(self.X_train_batch)
        return True

if __name__ == '__main__':
    class Opts(object):
        def __init__(self):
            self.window_size = 2
            self.vocab_size = 10
            #self.data_dir = '../data/text8'
            self.data_dir = '../data/dummy.txt'
            self.batch_size = 3 #4
    opts = Opts()
    data = Dataset(opts)
    fhand = open(opts.data_dir)
    all_data = data.tokenizer.texts_to_sequences(fhand)
    print(all_data)
    #print(all_data)
    #print(len(all_data))
    #incomplete = data.next_batch()
    #while incomplete:
    #    print(data.X_train_batch, data.y_train_batch) 
    #    incomplete = data.next_batch()
    count = 0
    print('epoch1')
    incomplete = data.next_batch()
    while incomplete:
        count+=1
        incomplete = data.next_batch()
    print(count)
    print(data.num_batches)
    print('epoch2')
    incomplete = data.next_batch()
    count = 0
    while incomplete:
        count+=1
        incomplete = data.next_batch()
    print(count)
    print(data.num_batches)
    print('epoch3')
    incomplete = data.next_batch()
    print('already started')
    count = 0
    while incomplete:
        count+=1
        incomplete = data.next_batch()
    print(count)
    print(data.num_batches)
