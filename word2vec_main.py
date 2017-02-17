from word2vec_model import run_model
import os
from argparse import ArgumentParser
import pickle
import sys

parser = ArgumentParser()

subparsers = parser.add_subparsers(title='different modes', dest='mode', description='train or test')
train_parser = subparsers.add_parser('train', help='train parsing')
train_parser.add_argument("-m", "--max_epochs", dest="max_epochs", help="max_epochs", type=int, default=5)
train_parser.add_argument("-d", "--data_dir",  dest="data_dir", help="data dir", default = '../data/clean_train')
train_parser.add_argument("-e", "--seed", dest="seed", help="set seed", type=int, default=0)
train_parser.add_argument("-w", "--widow_size", dest="window_size", help="window size", type=int, default=5)
train_parser.add_argument("-k", "--num_negative_samples", dest="num_negative_samples", help="num negative samples", type=int, default=5)
train_parser.add_argument("-E", "--embedding_dim", dest="embedding_dim", help="embedding dim", type=int, default=100)
train_parser.add_argument("-v", "--vocab_size", dest="vocab_size", help="vocab size", type=int, default=400000)
train_parser.add_argument("-b", "--batch_size", dest="batch_size", help="batch size", type=int, default=1000)
train_parser.add_argument("-r", "--lrate", dest="lrate", help="lrate", type=float, default=0.025)
train_parser.add_argument("-o", "--optimizer", dest="optimizer", help="name of optimizer", default="sgd")
train_parser.add_argument("-p", "--prob", dest="dropout_p", help="keep fraction", type=float, default=1.0)
test_parser = subparsers.add_parser('test', help='test parsing')

test_parser.add_argument("-d", "--model_dir", dest="model_dir", help="model directory")
test_parser.add_argument("-m", "--model_name", dest="modelname", help="model name")
test_parser.add_argument("-a", "--early_stopping", dest="early_stopping", help="early stopping", type=int, default=2)
test_parser.add_argument("-n", "--non_training", dest="non_training", help="non-training", type=int, default=1)

opts = parser.parse_args()

if opts.mode == "train":
    main_dir = '../word2vec_models/' + os.path.basename(opts.data_dir)

    model_dir = 'word2vec_embeddim{0}_seed{1}_dropout{2}_windowsize{3}_numsamples{4}_vocabsize{5}_lrate{6}_maxepochs{7}'.format(opts.embedding_dim, opts.seed, opts.dropout_p, opts.window_size, opts.num_negative_samples, opts.vocab_size, opts.lrate, opts.max_epochs)
    model_dir = os.path.join(main_dir, model_dir) 
    print(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    options_file = os.path.join(model_dir, 'options.pkl')
    with open(options_file, 'wb') as foptions:
        pickle.dump(opts, foptions)
        # if statement here. OPTS = best model
    run_model(opts, model_dir)

if opts.mode == "test":
    modelname = os.path.join(opts.model_dir, opts.modelname)

    with open(os.path.join(opts.model_dir, 'options.pkl'), 'rb') as foptions:
        options = pickle.load(foptions)
    # op_list = dir(options)
    # since we added new options, need to set them for old models
