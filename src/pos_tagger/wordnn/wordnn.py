import chainer
from chainer import cuda, functions as F, links as L, Variable, serializers
import numpy as np
import os
import sys
import yaml

sys.path.append('../')
from mean_max_threshold import zero_order_meanmax_filterloss as mm_filterloss


MODEL_CONFIG_NAME = 'model_config.yaml'


class WordnnTagger(chainer.Chain):
    def __init__(self, dim_embed, dim_hidden, n_vocab, n_label, window_size, objct='crossent', alpha=None):
        super(WordnnTagger, self).__init__(
            x2embed=L.EmbedID(n_vocab, dim_embed),
            embed2h=L.Linear(dim_embed * window_size, dim_hidden),
            h2y=L.Linear(dim_hidden, n_label)
        )
        self.dim_embed = dim_embed
        self.window_size = window_size
        assert (objct != 'mm_filterloss') or (alpha is not None)
        if objct == 'crossent':
            self.lossfunc = F.softmax_cross_entropy
        elif objct == 'mm_filterloss':
            self.lossfunc = mm_filterloss(alpha)
        else:
            raise Exception('Invalid objective : {}'.format(objct))

        # for saving model config
        self.model_config = {'dim_embed': dim_embed,
                             'dim_hidden': dim_hidden,
                             'n_vocab': n_vocab,
                             'n_label': n_label,
                             'window_size': window_size,
                             'objct': objct,
                             'alpha': alpha}

    def __call__(self, batch_xs, batch_ts):
        ts = Variable(batch_ts)
        ys = self.decode(batch_xs)
        return ys, self.lossfunc(ys, ts)

    def decode(self, batch_xs):
        batchsize = batch_xs.shape[0]
        all_embeds = self.x2embed(Variable(batch_xs))
        all_embeds = F.reshape(all_embeds, (batchsize, (self.window_size * self.dim_embed)))
        hs = self.embed2h(all_embeds)
        ys = self.h2y(F.tanh(hs))
        return ys

    # making oov vector by meaning all vectors
    def make_oov_vector(self, is_gpu=False):
        if is_gpu:
            oov_embed = cuda.cupy.array(cuda.cupy.sum(self.x2embed.W.data, axis=0) / len(self.x2embed.W.data), dtype=cuda.cupy.float32)
            self.x2embed.W.data[-1] = oov_embed
        else:
            oov_embed = np.array(np.sum(self.x2embed.W.data, axis=0) / len(self.x2embed.W.data), dtype=np.float32)
            self.x2embed.W.data[-1] = oov_embed

    @classmethod
    def load(clf, model_path):
        log_dir = os.path.dirname(model_path)
        config_path = os.path.join(log_dir, MODEL_CONFIG_NAME)
        with open(config_path) as f:
            model_config = yaml.load(f)
        vals = list(model_config.values())
        model = WordnnTagger(*vals)
        serializers.load_hdf5(model_path, model)
        return model

    def save(self, model_path):
        serializers.save_hdf5(model_path, self)

    def save_model_config(self, log_dir):
        config_path = os.path.join(log_dir, MODEL_CONFIG_NAME)
        with open(config_path, 'w') as fw:
            fw.write(yaml.dump(self.model_config))
