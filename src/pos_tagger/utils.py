import linecache
import numpy as np
import os
import yaml


SOS_token = '<s>'
EOS_token = '</s>'
UNK_token = '<UNK>'


# class MyArgs(object):
#     def __init__(self):
#         pass
#
#     def set_args(args_dict):
#         for


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    def get_word(self, i):
        return self.id2word[i]

    def add_special_token(self, tokens=[SOS_token, EOS_token, UNK_token]):
        for token in tokens:
            self.add(token)

    def check_words(self, words):
        new_words = []
        for word in words:
            if word in self.id2word:
                new_words.append(word)
            else:
                new_words.append(UNK_token)
        return new_words

    def load(self, data_path, lowercase_flg=False):
        with open(data_path) as f:
            if lowercase_flg:
                for word in f:
                    self.add(word.strip().lower())
            else:
                for word in f:
                    self.add(word.strip())


# data iterator from ntt data format
# line no % 3 == 1 : original sentence
# line no % 3 == 2 : POS tagging
# line no % 3 == 0 : blank line
def sentence_iter(data_path, n, rand_flg=True):
    n_line = sum(1 for _ in open(data_path))
    if rand_flg:
        line_idxs = np.random.permutation(n_line)
    else:
        line_idxs = [i for i in range(n_line)]
    line_idxs = [i+1 for i in line_idxs if i % 3 == 0]

    # generating
    sents = []
    pos_seqs = []
    for i, idx in enumerate(line_idxs):
        orig_sent = linecache.getline(data_path, idx)
        pos_seq = linecache.getline(data_path, idx+1)
        sents.append(orig_sent.strip())
        pos_seqs.append(pos_seq.strip())
        if (i+1) % n == 0:
            yield sents, pos_seqs
            sents = []
            pos_seqs = []


def line_iter(data_path, n, rand_flg=True):
    n_line = sum(1 for _ in open(data_path))
    if rand_flg:
        line_idxs = np.random.permutation(n_line)
    else:
        line_idxs = [i for i in range(n_line)]
    tags = []
    contexts = []
    for i, idx in enumerate(line_idxs):
        line = linecache.getline(data_path, idx+1)
        line = line.strip().split()
        tag, words = line[0], line[1:]
        tags.append(tag)
        contexts.append(words)
        if (i+1) % n == 0:
            yield [tags, contexts]
            tags = []
            contexts = []
    yield [tags, contexts]


def making_data(data_path, windowsize):
    data_file = os.path.basename(data_path)
    out_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    out_path = os.path.join(out_dir, data_file + '.{}window'.format(windowsize))

    if not os.path.exists(out_dir):
        os.system('mkdir {}'.format(out_dir))

    if os.path.exists(out_path):
        return out_path

    pad_size = windowsize // 2
    with open(out_path, 'w') as fw:
        for sent, pos_seq in sentence_iter(data_path, n=1, rand_flg=False):
            words = sent[0].split()
            tags = pos_seq[0].split()
            n_word = len(words)
            for _ in range(pad_size):
                words.insert(0, SOS_token)
                tags.insert(0, SOS_token)
                words.append(EOS_token)
                tags.append(EOS_token)
            for target_i in range(pad_size, n_word+pad_size):
                tag = tags[target_i]
                contexts = words[target_i-pad_size:target_i+pad_size+1]
                print(' '.join([tag] + contexts), file=fw)
    return out_path

# # dumping args in yaml format
# def args_dump(args, args_path):
#     with open(args_path, 'w') as fw:
#         fw.write(yaml.dump(vars(args)))
#
# def args_load(args_path):
