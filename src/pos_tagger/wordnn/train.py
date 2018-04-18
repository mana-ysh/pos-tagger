import argparse
from chainer import cuda, optimizer, optimizers
from datetime import datetime
import logging
import numpy as np
import os
import sys
import time

sys.path.append('../')
from utils import line_iter, making_data
from utils import Vocab
from wordnn import WordnnTagger


DIR_NAME = 'wordnn'


def train(args):
    if args.gpu > -1:
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np

    if args.log:
        log_dir = args.log
    else:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '{}_{}'.format(DIR_NAME, datetime.now().strftime('%Y%m%d_%H:%M')))

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # setting for logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(log_dir, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('Arguments...')
    for arg, val in vars(args).items():
        logger.info('{} : {}'.format(arg, val))

    logger.info('Loading Vocab...')
    vocab = Vocab()
    vocab.load(args.vocab)
    vocab.add_special_token()

    pos2id = Vocab()
    pos2id.load(args.poslist)

    logger.info('preparation for training data...')
    out_path = making_data(args.train_data, args.window)

    model = WordnnTagger(args.embed, args.hidden, len(vocab), len(pos2id), args.window, args.objct, args.alpha)
    model.save_model_config(log_dir)

    if args.gpu > -1:
        model.to_gpu()

    opt = getattr(optimizers, args.opt)()
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(args.gclip))
    opt.add_hook(optimizer.WeightDecay(args.wdecay))

    for epoch in range(args.epoch):
        logger.info('START epoch {}/{}'.format(epoch + 1, args.epoch))
        start = time.time()
        sum_loss = xp.zeros((), dtype=xp.float32)
        n_data = 0
        n_correct = 0
        for i, [tags, ctxts] in enumerate(line_iter(out_path, args.minibatch)):
            batch_ts = xp.array([pos2id[tag] for tag in tags], dtype=xp.int32)
            batch_xs = xp.array([[vocab[word] for word in context] for context in ctxts], dtype=xp.int32)
            cur_batch_size = batch_ts.shape[0]
            ys, loss = model(batch_xs, batch_ts)
            sum_loss += loss.data * cur_batch_size
            model.zerograds()
            loss.backward()
            opt.update()
            pred_labels = ys.data.argmax(1)
            n_correct += sum(1 for j in range(cur_batch_size) if pred_labels[j] == batch_ts[j])
            n_data += cur_batch_size
            logger.info('done {} batches'.format(i + 1))
        logger.info('{} epoch train loss = {}'.format(epoch + 1, sum_loss))
        logger.info('{} epoch train accuracy = {}'.format(epoch + 1, float(n_correct / n_data)))
        logger.info('{} sec for training per epoch'.format(time.time() - start))

        if args.valid_data:
            start = time.time()
            valid_loss, valid_accuracy = evaluation(model, args.valid_data, pos2id, vocab, args)
            logger.info('{} epoch valid loss = {}'.format(epoch + 1, valid_loss))
            logger.info('{} epoch valid accuracy = {}'.format(epoch + 1, valid_accuracy))
            logger.info('{} sec for validation per epoch'.format(time.time() - start))

        if args.test_data:
            start = time.time()
            test_loss, test_accuracy = evaluation(model, args.test_data, pos2id, vocab, args)
            logger.info('{} epoch test loss = {}'.format(epoch + 1, test_loss))
            logger.info('{} epoch test accuracy = {}'.format(epoch + 1, test_accuracy))
            logger.info('{} sec for testing per epoch'.format(time.time() - start))

        logger.info('serializing...')
        prefix = '{}_{}ep_{}embed_{}hidden_{}window_{}minibatch_{}opt'.format(DIR_NAME, epoch + 1, args.embed, args.hidden, args.window, args.minibatch, args.opt)
        model_path = os.path.join(log_dir, prefix + '.model')
        model.save(model_path)

    logger.info('done training')


def evaluation(model, data_path, pos2id, vocab, args):
    if args.gpu > -1:
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np
    model.make_oov_vector(args.gpu > -1)
    out_path = making_data(data_path, args.window)
    n_data = 0
    n_correct = 0
    sum_loss = xp.zeros((), dtype=xp.float32)
    for tags, contexts in line_iter(out_path, args.minibatch, False):
        batch_ts = xp.array([pos2id[tag] for tag in tags], dtype=xp.int32)
        batch_xs = xp.array([[vocab[word] for word in vocab.check_words(context)] for context in contexts], dtype=xp.int32)
        cur_batch_size = batch_ts.shape[0]
        ys, loss = model(batch_xs, batch_ts)
        sum_loss += loss.data * cur_batch_size
        pred_labels = ys.data.argmax(1)
        n_correct += sum(1 for j in range(cur_batch_size)
                         if pred_labels[j] == batch_ts[j])
        n_data += cur_batch_size
    accuracy = float(n_correct / n_data)
    return sum_loss, accuracy


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('training WordNN Tagger')
    argparser.add_argument('--log', default=None, help='output dir for log and models')
    argparser.add_argument('--vocab', default='../train02-21.wordlist', help='vocaburaly')
    argparser.add_argument('--poslist', default='../pentree.poslist', help='POS tag list')
    argparser.add_argument('--embed', type=int, default=200, help='dimention of word embedding')
    argparser.add_argument('--hidden', type=int, default=500, help='number of hidden units')
    argparser.add_argument('--train_data', help='training dataset path')
    argparser.add_argument('--valid_data', default=None, help='validation dataset path')
    argparser.add_argument('--test_data', default=None, help='test dataset path')
    argparser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    argparser.add_argument('--minibatch', type=int, default=256, help='mini-batch size')
    argparser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    argparser.add_argument('--window', type=int, default=5, help='window-size, must be odd')
    argparser.add_argument('--wdecay', type=float, default=0.0, help='weight decay param')
    argparser.add_argument('--opt', default='Adam', help='optimizer')
    argparser.add_argument('--gclip', default=5, help='gradient clipping')
    argparser.add_argument('--objct', default='crossent', help='loss function (crossent or mm_filterloss)')
    argparser.add_argument('--alpha', type=float, default=0.99, help='hyperparameter in mean-max threshold function')
    args = argparser.parse_args()

    train(args)
