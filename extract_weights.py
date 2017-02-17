import tensorflow as tf
import sys
import pickle

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph('{}.meta'.format(sys.argv[1]))

    with tf.Session() as sess:
	saver.restore(sess, sys.argv[1])

	print [var.name for var in tf.get_collection(tf.GraphKeys.VARIABLES)]
	#print [var.eval() for var in tf.get_collection(tf.GraphKeys.VARIABLES)]

	input_mat = tf.get_collection('input_mat')[0].eval()
    with open('../word2vec_models/text8/200000dictionary.pkl', 'rb') as fhand:
        dictionary = pickle.load(fhand) 
    reverse_dictionary = {v: k for k, v in dictionary.iteritems()}
    with open('vectors.txt', 'wt') as fhand:
        with open('vocabs.txt', 'wt') as fvocab:
            for i in xrange(input_mat.shape[0]):
                word = reverse_dictionary[i]
                fhand.write(' '.join([word]+map(str, list(input_mat[i]))))
                fhand.write('\n')
                fvocab.write(' '.join([word, str(i)]))
                fvocab.write('\n')
