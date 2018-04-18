import argparse
import chainer
from chainer import cuda, functions as F, links as L, optimizer, optimizers, Variable, serializers
from datetime import datetime
import logging
import numpy as np
import os
import sys
import time

sys.path.append('../')
from utils import line_iter, making_data
from utils import Vocab

from wordcsnn import WordCSnnTagger



def test(args):
    vocab = Vocab()
    vocab.load(args.vocab, args.lowercase)
    vocab.add_special_token()

    sufvocab = Vocab()
    sufvocab.load(args.sufvocab, args.lowercase)
    sufvocab.add_special_token(['s>', '<UNK>'])

    pos2id = Vocab()
    pos2id.load(args.poslist)

    if args.gpu > -1:
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np

    model = WordnnTagger.load(args.model)

    out_path = making_data(args.test_path, model.window)

    if args.gpu > -1:
        model.to_gpu()
    model.make_oov_vector(args.gpu > -1)

    # start evaluation
    n_data = 0
    n_correct = 0
    sum_loss = xp.zeros((), dtype=xp.float32)
    start = time.time()
    for tags, contexts in line_iter(out_path, args.minibatch, False):
        batch_ts = xp.array([pos2id[tag] for tag in tags], dtype=xp.int32)
        batch_caps = xp.array([[get_capf(word) for word in context] for context in contexts], dtype=xp.int32)
        if args.lowercase:
            contexts = [[word.lower() for word in context] for context in contexts]
        batch_xs = xp.array([[vocab[word] for word in vocab.check_words(context)] for context in contexts], dtype=xp.int32)
        # maybe inefficient...
        batch_sufs = [[word[-2:] for word in context] for context in contexts]
        batch_sufs = xp.array([[sufvocab[suf] for suf in sufvocab.check_words(sufs)] for sufs in batch_sufs], dtype=xp.int32)
        batch_caps = xp.array([[get_capf(word) for word in context] for context in contexts], dtype=xp.int32)
        batch_features = [batch_xs, batch_sufs, batch_caps]
        cur_batch_size = batch_ts.shape[0]
        ys, loss = model(batch_features, batch_ts)
        sum_loss += loss.data * cur_batch_size
        pred_labels = ys.data.argmax(1)
        n_correct += sum(1 for j in range(cur_batch_size) if pred_labels[j] == batch_ts[j])
        n_data += cur_batch_size
    end = time.time()
    accuracy = float(n_correct / n_data)
    print('test loss : {}'.format(sum_loss))
    print('test accuracy : {}'.format(accuracy))
    print('(time to run : {})'.format(end - start))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('testing FeedForwardNN Tagger using word feature')
    argparser.add_argument('--model', default=None, help='testing model')
    argparser.add_argument('--vocab', default='../train02-21.wordlist', help='vocaburaly')
    argparser.add_argument('--sufvocab', default='../train02-21.suflist', help='suffix vocaburary')
    argparser.add_argument('--poslist', default='../pentree.poslist', help='POS tag list')
    argparser.add_argument('--test_data', default=None, help='test dataset path')
    argparser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    argparser.add_argument('--minibatch', type=int, default=256, help='mini-batch size')
    argparser.add_argument('--lowercase', type=bool)
    args = argparser.parse_args()

    test(args)
