from __future__ import absolute_import
from __future__ import division

import string
import sys
import numpy as np
from six.moves import range
from six.moves import zip
from collections import deque

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f


def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()

    #text = text.translate(maketrans(filters, split*len(filters)))
    seq = text.split()
    return seq
    #return [_f for _f in seq if _f]


def one_hot(text, n, filters=base_filter(), lower=True, split=" "):
    seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]


class Tokenizer(object):
    def __init__(self, opts, nb_words=None, filters=base_filter(),
                 lower=True, split=' ', char_level=False):
        self.opts = opts
        self.word_counts = {}
        self.word_docs = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.nb_words = nb_words
        self.document_count = 0
        self.char_level = char_level
        self.words = []
        self.words_partition = []
        self.fhand = open(opts.data_dir, 'rt')
        self.previous_window = None
        self.previous_data = []
    def fit_on_texts(self, texts, non_split = False):
        '''Required before using texts_to_sequences or texts_to_matrix
        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        '''
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = text if self.char_level or non_split else text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        unk_count = sum([v[1] for v in wcounts[-(self.opts.vocab_size-1):]])
        self.count = [('UNK', unk_count)]
        self.count.extend(wcounts[:(self.opts.vocab_size-1)])
        sorted_voc = [wc[0] for wc in self.count]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(xrange(len(sorted_voc))))))


    def fit_on_sequences(self, sequences):
        '''Required before using sequences_to_matrix
        (if fit_on_texts was never called)
        '''
        self.document_count = len(sequences)
        self.index_docs = {}
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                if i not in self.index_docs:
                    self.index_docs[i] = 1
                else:
                    self.index_docs[i] += 1

    def texts_to_sequences(self, fhand):
        # this often takes too much memory
        data =[]
	for line in fhand:
            for word in line.split():
                if word in self.word_index:
                    index = self.word_index[word]
                else:
                    index = 0  # dictionary['UNK']
                data.append(index)
        return data
    def texts_to_sequences_partition(self, num_per_partition):
        # reading in the entire corpus is sometimes too much memory.
        # use this method in such cases.
        data = []
        for i, word in enumerate(self.words_partition):
            if word in self.word_index:
                index = self.word_index[word]
            else:
                index = 0  # dictionary['UNK']
            data.append(index)
            if len(data)==num_per_partition:
                # store the remaining in the line
                self.words_partition = self.words_partition[i+1:]
                previous_data = self.previous_data
                self.previous_data = data[-2*self.opts.window_size:]
                return previous_data + data

            # go to the next line
        for line in self.fhand:
            words = line.split()
            self.words_partition = words
            for i, word in enumerate(self.words_partition):
                if word in self.word_index:
                    index = self.word_index[word]
                else:
                    index = 0  # dictionary['UNK']
                data.append(index)
                if len(data)==num_per_partition:
                    # store the remaining in the line
                    self.words_partition = self.words_partition[i+1:]
                    previous_data = self.previous_data 
                    self.previous_data = data[-2*self.opts.window_size:]
                    return previous_data + data
        return None # no more lines of data



    def next_window(self):
        if self.previous_window is None:
            window = deque()
        else:
            window = self.previous_window
            window.popleft()
            # first use the remaining part of the previous line
        for i, word in enumerate(self.words):
            if word in self.word_index:
                index = self.word_index[word]
            else:
                index = 0  # dictionary['UNK']
            window.append(index)
            if len(window)==self.opts.window_size*2+1:
                # store the remaining 
                self.words = self.words[i+1:]
                self.previous_window = window
                return window
            # go to the next line

        for line in self.fhand:
            words = line.split()
            self.words = words
            for i, word in enumerate(words):
                if word in self.word_index:
                    index = self.word_index[word]
                else:
                    index = 0  # dictionary['UNK']
                window.append(index)
                if len(window)==self.opts.window_size*2+1:
                    self.previous_window = window
                    # store the remaining
                    self.words = self.words[i+1:]
                    return window
    def reset_window(self):
        self.fhand.close()
        self.fhand = open(self.opts.data_dir, 'rt')
        self.words = []
        self.words_partition = []
        self.previous_data = []
        self.previous_window = None

