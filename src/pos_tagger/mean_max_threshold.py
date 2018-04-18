import chainer
from chainer.utils import type_check
import numpy as np


class ZeroOrderMeanmaxFilterloss(chainer.Function):
    """
    Hinge loss of mean max threshold function [Weiss 2014]
    Detail formulation is in my note.
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.int32,
            in_types[0].ndim == in_types[1].ndim + 1,
            in_types[0].shape[0] == in_types[1].shape[0],
            in_types[0].shape[2:] == in_types[1].shape[1:])

    def forward(self, inputs):
        """
        x : score matrix (2d : batch-size x n_tag)
        t : GOLD tags (1d : batch-size)
        """
        x, t = inputs
        num = x.dtype.type(len(x))
        max_score = x.max(axis=1)  # (1d : batch-size)
        gold_score = x[np.arange(x.shape[0]), t]
        mean_score = x.mean(axis=1)
        mm_threshold = self.alpha * max_score + (1. - self.alpha) * mean_score
        margin = 1. + mm_threshold - gold_score
        self.bottom_diff = np.maximum(np.zeros(margin.shape), margin)
        loss = self.bottom_diff.sum() / num
        return np.array(loss, dtype=x.dtype),

    def backward(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        num = x.dtype.type(len(x))
        max_idxs = x.argmax(axis=1)
        k = x.shape[1]
        gx = np.repeat(self.bottom_diff, x.shape[1]).reshape(x.shape)
        mask_gx = np.sign(gx)
        gx = np.sign(gx) * ((1 - self.alpha) / k) * gloss
        gx[np.arange(x.shape[0]), t] -= 1
        gx[np.arange(x.shape[0]), max_idxs] += self.alpha
        gx *= mask_gx
        return gx.astype(np.float32), None


# maybe inefficient...
class zero_order_meanmax_filterloss():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x, t):
        return ZeroOrderMeanmaxFilterloss(self.alpha)(x, t)


def meanmax_filtering(ys, alpha):
    pred_tags = []
    for i in range(len(ys)):
        max_score = ys[i].max()
        mean_score = ys[i].mean()
        mm_threshold = alpha * max_score + (1. - alpha) * mean_score
        score = np.maximum(ys[i] - mm_threshold, 0)
        tag_cands = [i for i in score.argsort()[::-1] if score[i] > 0]
        assert len(tag_cands) > 0
        pred_tags.append(tag_cands)
    return pred_tags
