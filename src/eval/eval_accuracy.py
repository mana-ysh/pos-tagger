import argparse


# assuming NTT format
def cal_accuracy(gold_file, pred_file, k=1):
    n_data = 0
    n_correct = 0
    with open(gold_file) as fg, open(pred_file) as fp:
        for i, (g_line, p_line) in enumerate(zip(fg, fp)):
            if i % 3 == 1:
                assert len(g_line.split()) == len(p_line.split())
                g_tags = g_line.strip().split()
                p_tags = p_line.strip().split()
                p_tags = [tag_cands.split('@')[:k] for tag_cands in p_tags]
            else:
                continue
            n_data += len(g_tags)
            n_correct += sum(1 for g, p in zip(g_tags, p_tags) if g in p)
    print('{}-best Accuracy = {} / {} = {}'.format(k, n_correct, n_data, float(n_correct/n_data)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gold')
    argparser.add_argument('--pred')
    argparser.add_argument('--kbest', default=1, type=int)
    args = argparser.parse_args()
    cal_accuracy(args.gold, args.pred, args.kbest)
