
"""
TODO
"""

import argparse
import numpy as np
import os
import pickle
import re
import time

from lattice import Lattice
from utils import sent_tag_iter
from hocrf import HigherOrderCRF


posdict_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/pos_dict.pkl'
vocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train.02-21.sent.vocab.pkl'
sufvocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train.02-21.sent.suf_vocab.pkl'


UNK_token = '<UNK>'
NUMBER_OF_POS = 45
HIDDEN = 500

with open(posdict_path, 'rb') as f:
    pos2id = pickle.load(f)
pos2id['<s>'] = len(pos2id)
pos2id['</s>'] = len(pos2id)
pos2id['<UNK>'] = len(pos2id)

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab['<s>'] = len(vocab)
vocab['</s>'] = len(vocab)
vocab['<UNK>'] = len(vocab)
with open(sufvocab_path, 'rb') as f:
    sufvocab = pickle.load(f)
sufvocab['s>'] = len(sufvocab)
sufvocab['<UNK>'] = len(sufvocab)


def train(args):
    # n_feature = 1 + (NUMBER_OF_POS + 2) ** (args.pos_lorder + args.pos_rorder + 1)
    n_feature = (HIDDEN + NUMBER_OF_POS) * NUMBER_OF_POS
    print('building CRF...')
    model = HigherOrderCRF.build(n_feature, args.pos_lorder, args.pos_rorder, args.posmodel, args.hoposmodel, args.lr, args.train_alpha, args.l2, args.test_alpha)
    model.init_weight()
    print('build lattice')
    l = Lattice(args.pos_lorder, args.pos_rorder, model)
    print('done')
    for epoch in range(args.epoch):
        print('Start epoch {}'.format(epoch + 1))
        s = time.time()
        n_word = 0
        n_corr = 0
        sum_log_ll = 0
        for [words, postags] in sent_tag_iter(args.train_data):
            postags = [pos2id[pos] for pos in postags]
            sufs = [sufvocab[suf] for suf in check_sufvocab(words)]
            caps = [word2capf(word) for word in words]
            words = [vocab[word] for word in check_vocab(words)]
            features = [words, sufs, caps]
            predtags = l.viterbe_decode_hocrf(features, train_flg=True)
            assert len(postags) == len(predtags)
            n_corr += sum(1 for i in range(len(postags)) if postags[i] == predtags[i])
            n_word += len(words)
            log_ll = l.update_hocrf(features, postags)
            sum_log_ll += log_ll
        print('Train accuracy : {} / {} = {}'.format(n_corr, n_word, float(n_corr/n_word)))
        print('Train Log Likelifood : {}'.format(sum_log_ll))
        print('Running Time in this epoch : {} sec'.format(time.time() - s))

        if args.valid_data:
            valid_accuracy = validation(l, args.valid_data)
            print('Valid accuracy : {}'.format(valid_accuracy))

        if args.test_data:
            test_accuracy = validation(l, args.test_data)
            print('Test accuracy : {}'.format(test_accuracy))


def validation(l, valid_data):
    n_word = 0
    n_corr = 0
    for [words, postags] in sent_tag_iter(valid_data, False):
        postags = [pos2id[pos] for pos in postags]
        sufs = [sufvocab[suf] for suf in check_sufvocab(words)]
        caps = [word2capf(word) for word in words]
        words = [vocab[word] for word in check_vocab(words)]
        features = [words, sufs, caps]
        predtags = l.viterbe_decode_hocrf(features, train_flg=False)
        assert len(postags) == len(predtags)
        n_corr += sum(1 for i in range(len(postags)) if postags[i] == predtags[i])
        n_word += len(words)
    accuracy = float(n_corr/n_word)
    return accuracy




def check_vocab(words):
    new_words = []
    for word in words:
        if word in vocab:
            new_words.append(word)
        else:
            new_words.append(UNK_token)
    return new_words


def check_sufvocab(words):
    new_sufs = []
    for word in words:
        if word[-2:] in sufvocab:
            new_sufs.append(word[-2:])
        else:
            new_sufs.append(UNK_token)
    return new_sufs


def word2capf(word):
    if word.islower():
        return 0
    elif word[0].isupper():
        return 1
    elif word.isupper():
        return 2
    elif re.search(r'[A-Z]', word):
        return 3
    else:
        return 4


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--posmodel', help='POS Tagger')
    p.add_argument('--hoposmodel', help='Higher Order POS Tagger')
    p.add_argument('--train_data', help='POS tag paired with sentence data for training')
    p.add_argument('--valid_data', default=None, help='POS tag paired with sentence data for validation')
    p.add_argument('--test_data', default=None, help='POS tag paired with sentence data for testing')
    p.add_argument('--pos_lorder', type=int, help='number of left context order in lattice')
    p.add_argument('--pos_rorder', type=int, help='number of right context order in lattice')
    p.add_argument('--train_alpha', type=float, help='hyperparameter in mean-max threshold function (training)')
    p.add_argument('--test_alpha', type=float, help='hyperparameter in mean-max threshold function (test)')
    p.add_argument('--lr', type=float, help='learning rate')
    p.add_argument('--epoch', type=int, help='number of epoch')
    p.add_argument('--l2', type=float, default=0., help='hyperparameter in L2 regularization')

    train(p.parse_args())
