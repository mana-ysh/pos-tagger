
import linecache
import numpy as np


def sent_tag_iter(data_path, rand_flg=True):
    n_line = sum(1 for _ in open(data_path)) // 3  # one sample have three line
    print(n_line)
    if rand_flg:
        line_idxs = np.random.permutation(n_line)
    else:
        line_idxs = [i for i in range(n_line)]
    sent = ''
    tags = ''
    for idx in line_idxs:
        sent = linecache.getline(data_path, 3*idx+1).strip().split()
        tags = linecache.getline(data_path, 3*idx+2).strip().split()
        assert len(sent) == len(tags)
        yield [sent, tags]
