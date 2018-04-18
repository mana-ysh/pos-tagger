# implementation of higher order CRF
"""
TODO
"""

from chainer import serializers
import copy
import math
import numpy as np
import os
import pickle
import sys

sys.path.append('../')
# from ffnn_pos2 import FeedForwardNN as POSTagger
# from ffnn_pos3_wdecay import FeedForwardNN as HOPOSTagger
from wordcsnn.wordcsnn import WordCSnnTagger
from wordcsnn.howordcsnn import HOWordCSnnTagger

NUMBER_OF_POS = 45

HIDDEN = 500
WEMBED = 200
FEMBED = 20
PEMBED = 50
WINDOW = 5
L_POS_ORDER = 2
R_POS_ORDER = 2


vocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train.02-21.sent.vocab.pkl'
sufvocab_path = os.path.abspath(os.path.dirname(__file__)) + '/../../data/train.02-21.sent.suf_vocab.pkl'

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab['<s>'] = len(vocab)
vocab['</s>'] = len(vocab)
vocab['<UNK>'] = len(vocab)
with open(sufvocab_path, 'rb') as f:
    sufvocab = pickle.load(f)
sufvocab['s>'] = len(sufvocab)
sufvocab['<UNK>'] = len(sufvocab)


class HigherOrderCRF(object):
    def __init__(self, n_feature, left_order, right_order, pos_model, hopos_model, lr, train_alpha, l2_reg, test_alpha):
        self.n_feature = n_feature
        self.left_order = left_order
        self.right_order = right_order
        self.lr = lr
        self.pos_model = pos_model
        self.hopos_model = hopos_model
        self.train_alpha = train_alpha
        self.l2_reg = l2_reg
        # self.k = k
        self.test_alpha = test_alpha

    def init_weight(self):
        self.w = np.zeros((self.n_feature, ), dtype=np.float64)

    def update(self, gold_feature, expect_feature):
        delta_w = (gold_feature - expect_feature) - (self.l2_reg * self.w)
        self.w = self.w + self.lr * delta_w

    def cache_pos_score(self, features):
        _features = copy.deepcopy(features)
        n_word = len(_features[0])
        batch_words, batch_sufs, batch_caps = make_feature_batch(_features, WINDOW)
        self.batch_features = [batch_words, batch_sufs, batch_caps]
        # caching POS scores
        pos_ys, _ = self.pos_model.forward_batch(self.batch_features, np.zeros((n_word, ), dtype=np.int32))  # TODO : dealing with dummy batch_ts
        assert pos_ys.data.shape == (n_word, NUMBER_OF_POS)
        self.pos_scores = pos_ys.data[:]

    def cache_hopos_feature_vector(self, state_infos):
        batch_words = []
        batch_sufs = []
        batch_caps = []
        batch_poss = []
        for s_info in state_infos:
            batch_words.append(self.batch_features[0][s_info['t_step']-1])
            batch_sufs.append(self.batch_features[1][s_info['t_step']-1])
            batch_caps.append(self.batch_features[2][s_info['t_step']-1])
            batch_poss.append(s_info['pos_context'])
        batch_words = np.array(batch_words, dtype=np.int32)
        batch_sufs = np.array(batch_sufs, dtype=np.int32)
        batch_caps = np.array(batch_caps, dtype=np.int32)
        batch_poss = np.array(batch_poss, dtype=np.int32)
        hopos_features = self.hopos_model.get_hopos_feature([batch_words, batch_sufs, batch_caps, batch_poss])
        self.hopos_features = hopos_features.data[:]

    def pos_score(self, t_step, pos_id):
        assert t_step > 0
        return self.pos_scores[t_step-1][pos_id]

    def hopos_feature(self, idx):
        return self.hopos_features[idx]

    def mean_max_flt(self, gold_tags=None, train_flg=True):
        """
        Filtering POS tags by using mean-max threshold
        """
        if train_flg:
            alpha = self.train_alpha
        else:
            alpha = self.test_alpha
        max_score = self.pos_scores.max(axis=1)
        mean_score = self.pos_scores.mean(axis=1)
        threshold = alpha * max_score + (1. - alpha) * mean_score
        flt_pos_list = [[i for i in range(self.pos_scores.shape[1])
                        if self.pos_scores[j][i] >= threshold[j]]
                        for j in range(self.pos_scores.shape[0])]
        # In training, GOLD tag must not be pruned
        if gold_tags:
            assert len(gold_tags) == len(flt_pos_list)
            for i in range(len(gold_tags)):
                if gold_tags[i] not in flt_pos_list[i]:
                    flt_pos_list[i].append(gold_tags[i])
        return flt_pos_list

    def kbest_flt(self, gold_tags=None):
        raise NotImplementedError

    def cal_log_likelihood(self, gold_feature, log_z):
        score = self.w.dot(gold_feature)
        return score - log_z

    @classmethod
    def build(cls, n_feature, left_order, right_order, pos_model_path, hopos_model_path, lr, train_alpha, l2_reg, test_alpha):
        pos_model = WordCSnnTagger(WEMBED, FEMBED, HIDDEN, len(vocab), NUMBER_OF_POS, len(sufvocab), WINDOW)
        hopos_model = HOWordCSnnTagger(WEMBED, FEMBED, PEMBED, HIDDEN, len(vocab), NUMBER_OF_POS, len(sufvocab), WINDOW, L_POS_ORDER, R_POS_ORDER)
        serializers.load_hdf5(pos_model_path, pos_model)
        serializers.load_hdf5(hopos_model_path, hopos_model)
        assert left_order == L_POS_ORDER
        assert right_order == R_POS_ORDER
        return HigherOrderCRF(n_feature, left_order, right_order, pos_model, hopos_model, lr, train_alpha, l2_reg, test_alpha)


def make_feature_batch(fs, window_size):
    """
    making one sentence into number of words size batch
    and preprocessing for NN model
    Asumming
    - words is already replaced with ids
    - already padding ?
    """
    n_context = window_size // 2
    words, sufs, caps = fs
    for _ in range(n_context):
        words.insert(0, vocab['<s>'])
        words.append(vocab['</s>'])
        sufs.insert(0, sufvocab['s>'])
        sufs.append(sufvocab['s>'])
        caps.insert(0, 0)  # CAUTION! cap feature in SOS and EOS token is 0
        caps.append(0)  # CAUTION!
    batch_words = [words[i-n_context:i+n_context+1] for i in range(n_context, len(words)-n_context)]
    batch_sufs = [sufs[i-n_context:i+n_context+1] for i in range(n_context, len(sufs)-n_context)]
    batch_caps = [caps[i-n_context:i+n_context+1] for i in range(n_context, len(caps)-n_context)]
    return np.array(batch_words, dtype=np.int32), np.array(batch_sufs, dtype=np.int32), np.array(batch_caps, dtype=np.int32)
