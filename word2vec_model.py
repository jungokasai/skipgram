# Tensorflow Implementation of Negative Sampling. Noise Constrastive Estimation is provided by TensorFlow, but Negative Sampling is not
## Implemented by Jungo Kasai 02/09/2017


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from utils.data_process import Dataset
import tensorflow as tf


class Word2Vec(object):

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[self.opts.batch_size])
        self.labels_placeholder = tf.placeholder(tf.int64, shape=[self.opts.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)

    def add_input_embeddings(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('input_mat'):
                embedding_mat = tf.get_variable('embedding_mat', initializer=tf.random_uniform([self.opts.vocab_size, self.opts.embedding_dim], -1.0, 1.0))
            tf.add_to_collection('input_mat', embedding_mat)
            self.input_embeddings = tf.nn.embedding_lookup(embedding_mat, self.input_placeholder)  # batch_size by embedding_dim
            

    def add_true_embeddings(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('output_mat'):
                self.embedding_mat = tf.get_variable('embedding_mat', initializer=tf.random_uniform([self.opts.vocab_size, self.opts.embedding_dim], -1.0, 1.0))
            self.true_embeddings = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding_mat, self.labels_placeholder), self.keep_prob)  # batch_size by embedding_dim

    def add_negative_embeddings(self):
        negative_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(tf.expand_dims(self.labels_placeholder, -1), 1, self.opts.batch_size, False, self.opts.vocab_size, unigrams=self.loader.unigrams)

        with tf.device('/cpu:0'):
            embed_vectors = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding_mat, negative_samples), self.keep_prob)  # batch_size by embedding_dim
        return embed_vectors

    def add_positive_loss(self):
        batch_output = tf.log(tf.nn.sigmoid(tf.reduce_sum(self.input_embeddings * self.true_embeddings, -1)))
        loss = -tf.reduce_mean(batch_output)
        return loss

    def add_negative_loss(self):
        embed_vectors = self.add_negative_embeddings()
        batch_output = tf.log(tf.nn.sigmoid(-tf.reduce_sum(self.input_embeddings * embed_vectors, -1)))  # notice the minus sign
        loss = -tf.reduce_mean(batch_output)
        return loss

    def add_training_op(self, loss):
        with tf.device('/cpu:0'):
            if self.opts.optimizer == 'sgd':
                tot_num_batches = self.loader.num_batches*self.opts.max_epochs 
                values = list(np.arange(self.opts.lrate, 0.0, -0.001))
                boundaries = list(np.arange(tot_num_batches//len(values), tot_num_batches, tot_num_batches//len(values)).astype('int32'))
                learning_rate = tf.train.piecewise_constant(self.loader._epoch_completed+self.loader._batch_index, boundaries, values)
                
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.opts.lrate)
            elif self.opts.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.opts.lrate)
            else:
                raise ValueError('improper optimizer_name {}'.format(self.opts.optimizer))
            train_op = optimizer.minimize(loss)
            return train_op

    def run_batch(self, session):
        feed = {}
        feed[self.input_placeholder] = self.loader.X_train_batch
        feed[self.labels_placeholder] = self.loader.y_train_batch
        feed[self.keep_prob] = self.opts.dropout_p
        train_op = self.train_op
        loss, _ = session.run([self.loss, self.train_op], feed_dict=feed)
        return loss

    def run_epoch(self, session, saver, saving_dir):
        epoch_start_time = time.time()
	epoch_incomplete = self.loader.next_batch()
        while epoch_incomplete:
            loss = self.run_batch(session)
            #self.train_loss.append(loss)
            print('{}/{}, loss {:.4f}'.format(self.loader._batch_index, self.loader.num_batches, loss), end = '\r') 
            epoch_incomplete = self.loader.next_batch()
            if self.loader._batch_index % (self.loader.num_batches//100) == 0:
                saving_file = os.path.join(saving_dir, 'epoch{:.2f}_loss{:.4f}'.format(self.loader._epoch_completed+self.loader._batch_index/float(self.loader.num_batches), loss))
                saver.save(session, saving_file)
                print('saving it to {}'.format(saving_file))
        epoch_end_time = time.time()

        epoch_end_time = time.time()
        m, s = divmod(epoch_end_time - epoch_start_time, 60)
        h, m = divmod(m, 60)
        print('Epoch Time: {0}:{1}:{2}'.format(int(h), int(m), s))
        return loss

    def __init__(self, opts):

        self.opts = opts
        self.loader = Dataset(opts)
        self.add_placeholders()
        self.add_input_embeddings()
        self.add_true_embeddings()
        loss = self.add_positive_loss()
        for i in xrange(self.opts.num_negative_samples):
            loss += self.add_negative_loss()
        self.loss = loss
        self.train_op = self.add_training_op(loss)
        self.train_loss = []


def run_model(opts, saving_dir, loader=None, modelname=None):
    g = tf.Graph()
    with g.as_default():
        if not modelname:  # do not re-seed for further training option
            tf.set_random_seed(opts.seed)
        model = Word2Vec(opts)
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            if modelname:
                print('using an existing model')
                saver = tf.train.Saver()
                saver.restore(session, modelname)

            for i in xrange(opts.max_epochs):
                print('Epoch {}'.format(i + 1))
                loss = model.run_epoch(session, saver, saving_dir)
                saving_file = os.path.join(saving_dir, 'epoch{0}_loss{1:.5f}.weights'.format(i + 1, loss))
                print('saving it to {}'.format(saving_file))
                saver.save(session, saving_file)
