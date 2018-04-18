
import copy
from graphviz import Digraph
from itertools import product
import numpy as np

from hocrf import HigherOrderCRF

# import warnings
# warnings.filterwarnings("error")

NUMBER_OF_POS = 45

HIDDEN = 500
WEMBED = 200
FEMBED = 20
PEMBED = 50
WINDOW = 5

START_TAG_ID = 45
END_TAG_ID = 46
INIT_STATE_LABEL = -1
END_STATE_LABEL = -2


class Lattice(object):
    """
    expanding 2n-order lattice for supertagging pipeline
    """
    def __init__(self, left_n, right_n, model):
        assert left_n > -1 and right_n > -1
        self.left_n = left_n
        self.right_n = right_n
        self.model = model
        self.state_mat = None
        self.label2id = {str(INIT_STATE_LABEL): 0, str(END_STATE_LABEL): 1}  # for visualization
        self.state2id = {}
        pos_ids = [str(i) for i in range(NUMBER_OF_POS+2)]
        # self.pos_seq2id = {' '.join(pos_seq): i for i, pos_seq in enumerate(product(pos_ids, repeat=left_n+right_n+1))}  # for HOCRF

    def reset_lattice(self):
        self.state_mat = None
        self.label2id = {str(INIT_STATE_LABEL): 0, str(END_STATE_LABEL): 1}

    def expand(self, words, tags, gold_tags=None):
        self.reset_lattice()
        self.words = words
        self.tags = tags
        n_word = len(words)
        assert n_word == len(tags)

        for _ in range(self.left_n):
            tags.insert(0, [START_TAG_ID])
        for _ in range(self.right_n):
            tags.append([END_TAG_ID])
        tags.append([END_TAG_ID])

        self.state_mat = []  # assuming len(state_mat) is equal to (sentence length + 2)
        self.state_mat.append([State([INIT_STATE_LABEL], self.left_n, self.right_n, t_step=0)])
        for t, i in enumerate(range(self.left_n, self.left_n + n_word)):
            cur_t = tags[i]
            prev_ts = tags[i-self.left_n:i]
            next_ts = tags[i+1:i+self.right_n+1]

            state_cands = []
            # enumetaring all state candidates in this timestep
            for state_label in product(*prev_ts, cur_t, *next_ts):
                state_cands.append(State(list(state_label), self.left_n, self.right_n, t_step=t+1))
                # state_cands.append(state_label)
                if state_label not in self.label2id:
                    # self.label2id[state_label] = len(self.label2id)
                    self.label2id[' '.join(map(str, state_label))] = len(self.label2id)
            self.state_mat.append(state_cands)
        self.state_mat.append([State([END_STATE_LABEL], self.left_n, self.right_n, t_step=n_word+1)])

        # connecting by edge
        self.edges = []
        # adding connection from initial state
        prev_states, cur_states = self.state_mat[0], self.state_mat[1]
        for prev_s in prev_states:
            for cur_s in cur_states:
                prev_sname = '0' + '-' + str(self.label2id[prev_s.get_label()])
                cur_sname = '1' + '-' + str(self.label2id[cur_s.get_label()])
                self.edges.append([prev_sname, cur_sname])
                prev_s.add_next_state(cur_s)
                cur_s.add_prev_state(prev_s)

        prev_states = cur_states
        # adding main connections
        for i in range(2, n_word+1):
            cur_states = self.state_mat[i]
            for prev_s in prev_states:
                for cur_s in cur_states:
                    if connect_judge(prev_s, cur_s):
                        prev_sname = str(i-1) + '-' + str(self.label2id[prev_s.get_label()])
                        cur_sname = str(i) + '-' + str(self.label2id[cur_s.get_label()])
                        self.edges.append([prev_sname, cur_sname])
                        prev_s.add_next_state(cur_s)
                        cur_s.add_prev_state(prev_s)
            prev_states = cur_states
        # adding connections to end state
        cur_states = self.state_mat[n_word+1]
        for prev_s in prev_states:
            for cur_s in cur_states:
                prev_sname = str(i) + '-' + str(self.label2id[prev_s.get_label()])
                cur_sname = str(i+1) + '-' + str(self.label2id[cur_s.get_label()])
                self.edges.append([prev_sname, cur_sname])
                prev_s.add_next_state(cur_s)
                cur_s.add_prev_state(prev_s)

        # setting state id (assigned to each state uniquely) and info TODO: inefficient. this loop can be integrated other loop
        cnt = 0
        self.state_infos = []
        for states in self.state_mat[1:-1]:
            for s in states:
                s.state_id = cnt
                self.state_infos.append({'t_step': s.t_step,
                                         'pos_context': s.leftpos + s.rightpos})
                cnt += 1

    # visualization of lattice by graphviz
    def visualize(self, sptag_flg=False):
        dot = Digraph(comment='Lattice')
        dot.graph_attr['rankdir'] = 'LR'
        dot.body.append(r'label = "\nwords : {}\ntag candidates : {}"'.format(self.words, self.tags[self.left_n:-self.right_n-1]))
        # adding nodes
        for i, states in enumerate(self.state_mat):
            for s in states:
                sname = str(i) + '-' + str(self.label2id[s.get_label()])
                if sptag_flg:
                    if (i == 0) or (i == len(self.state_mat) - 1):
                        dot.node(sname, str(s.get_label()) + '\nalpha={:.2f}'.format(s.alpha))
                    else:
                        dot.node(sname, str(s.get_label()) + '\nalpha={:.2f}\nPOS score={:.2f}\nSP score={:.2f}'.format(s.alpha, self.model.pos_score(i, s.curpos), s.sptag_score))
                else:
                    if (i == 0) or (i == len(self.state_mat) - 1):
                        # dot.node(sname, str(s.get_label()) + '\nalpha={:.2f}'.format(s.alpha))
                        dot.node(sname, str(s.get_label()) + '\nalpha={:.6f}\nbeta={:.6f}'.format(s.fw_alpha, s.fw_beta))
                    else:
                        # dot.node(sname, str(s.get_label()) + '\nalpha={:.2f}\nPOS score={:.2f}'.format(s.alpha, self.model.pos_score(i, s.curpos)))
                        dot.node(sname, str(s.get_label()) + '\nalpha={:.6f}\nbeta={:.6f}'.format(s.fw_alpha, s.fw_beta))
        # adding edges
        for e in self.edges:
            dot.edge(e[0], e[1])
        dot.render('lattice')
        del dot

    def viterbe_decode(self, features, filter_flg=True):
        words = features[0]
        # print(words)
        self.model.cache_pos_score(features)
        flt_pos_list = self.model.mean_max_flt()
        # print(words)
        # print(flt_pos_list)
        self.expand(words, flt_pos_list)

        # self.visualize()

        state_infos = []
        # print(len(self.state_mat))
        for i, states in enumerate(self.state_mat):
            # skipping init and end state
            if i == 0 or i == len(self.state_mat) - 1:
                continue
            for s in states:
                state_infos.append([i] + s.pos_list)
        self.model.cache_sp_score(state_infos)
        # starting viterbe
        for i in range(1, len(self.state_mat)-1):
            prev_states = self.state_mat[i-1]
            # cur_states = self.state_mat[i]
            for prev_s in prev_states:
                for cur_state in prev_s.get_next_states():
                    # TODO: edge score cannot be considered?????
                    sptag_scores = self.model.all_sptag_scores(i, cur_state.pos_list)
                    sptag_score = sptag_scores.max()
                    cur_state.sptag_score = sptag_score
                    cur_state.sptag = sptag_scores.argmax()  # set supertag to state
                    score = prev_s.alpha + self.model.pos_score(i, cur_state.curpos) + sptag_score
                    # score = prev_s.alpha + self.model.pos_score(i, cur_state.curpos)
                    # score = self.model.pos_score(i, cur_state.curpos)  # 間違い
                    if score > cur_state.alpha:
                        cur_state.alpha = score
                        cur_state.bp = prev_s
        # for end state
        prev_states = self.state_mat[i]
        for prev_s in prev_states:
            assert len(prev_s.get_next_states()) == 1
            for end_state in prev_s.get_next_states():
                score = prev_s.alpha
                if score > end_state.alpha:
                    end_state.alpha = score
                    end_state.bp = prev_s

        # backtrace
        best_poss = []
        best_sptags = []

        bp_state = self.state_mat[-1][0]  # this is end state
        while True:
            bp_state = bp_state.bp
            if bp_state.bp is None:
                break
            # print(bp_state.pos_list)
            best_poss.append(bp_state.curpos)
            best_sptags.append(bp_state.sptag)
        best_poss.reverse()
        best_sptags.reverse()
        return best_poss, best_sptags

    def forward_backward(self, features, gold_tags):
        self.model.cache_pos_score(features)
        flt_pos_list = self.model.mean_max_flt(gold_tags=gold_tags)
        words = features[0]
        self.expand(words, flt_pos_list)
        self.model.cache_hopos_feature_vector(self.state_infos)
        # setting feature vector and gold_flg in each state
        _gold_tags = copy.deepcopy(gold_tags)
        for _ in range(self.left_n):
            _gold_tags.insert(0, START_TAG_ID)
        for _ in range(self.right_n):
            _gold_tags.append(END_TAG_ID)
        _gold_tags.append(END_TAG_ID)
        for i in range(1, len(self.state_mat)-1):
            for state in self.state_mat[i]:
                # setting feature vector
                self.build_feature_vector2(state)
                phi = np.exp(np.dot(state.feature_vector, self.model.w))
                state.set_phi(phi)
                # setting GOLD Flag
                if state.pos_list == _gold_tags[i-1:i+self.left_n+self.right_n]:
                    state.gold_flg = True

        # forward
        for i in range(1, len(self.state_mat)):
            for cur_state in self.state_mat[i]:
                max_prev_log_alpha = np.float64('-inf')
                for prev_s in cur_state.get_prev_states():
                    if max_prev_log_alpha < prev_s.fw_log_alpha:
                        max_prev_log_alpha = prev_s.fw_log_alpha
                sumexp = np.float64(0.)
                for prev_s in cur_state.get_prev_states():
                    sumexp += np.exp(prev_s.fw_log_alpha - max_prev_log_alpha) * prev_s.phi
                cur_state.fw_log_alpha = max_prev_log_alpha + np.log(sumexp)

        # backward
        for i in reversed(range(0, len(self.state_mat)-1)):
            for cur_state in self.state_mat[i]:
                max_next_log_beta = np.float64('-inf')
                for next_s in cur_state.get_next_states():
                    if max_next_log_beta < next_s.fw_log_beta:
                        max_next_log_beta = next_s.fw_log_beta
                sumexp = np.float64(0.)
                for next_s in cur_state.get_next_states():
                    sumexp += np.exp(next_s.fw_log_beta - max_next_log_beta) * next_s.phi
                cur_state.fw_log_beta = max_next_log_beta + np.log(sumexp)
        self.log_norm_term = self.state_mat[-1][0].fw_log_alpha

    def update_hocrf(self, features, gold_tags):
        self.forward_backward(features, gold_tags)
        dim_vector = self.model.n_feature
        gold_feature = np.zeros((dim_vector, ), dtype=np.float64)
        expect_feature = np.zeros((dim_vector, ), dtype=np.float64)
        for states in self.state_mat[1:-1]:
            for s in states:
                log_marginal_prob = s.fw_log_alpha + s.fw_log_beta - self.log_norm_term + np.log(s.phi)
                marginal_prob = np.exp(log_marginal_prob)
                expect_feature += marginal_prob * s.feature_vector
                if s.gold_flg:
                    gold_feature += s.feature_vector
        log_ll = self.model.cal_log_likelihood(gold_feature, self.log_norm_term)
        self.model.update(gold_feature, expect_feature)
        return log_ll

    def viterbe_decode_hocrf(self, features, train_flg):
        words = features[0]
        self.model.cache_pos_score(features)
        flt_pos_list = self.model.mean_max_flt(train_flg=train_flg)
        self.expand(words, flt_pos_list)
        self.model.cache_hopos_feature_vector(self.state_infos)
        for i in range(1, len(self.state_mat)-1):
            for state in self.state_mat[i]:
                self.build_feature_vector2(state)
                log_phi = np.dot(state.feature_vector, self.model.w)
                state.score = log_phi

        # starting viterbe
        for i in range(1, len(self.state_mat)-1):
            prev_states = self.state_mat[i-1]
            for prev_s in prev_states:
                for cur_state in prev_s.get_next_states():
                    score = prev_s.alpha + cur_state.score
                    if score > cur_state.alpha:
                        cur_state.alpha = score
                        cur_state.bp = prev_s
        # for end state
        prev_states = self.state_mat[i]
        for prev_s in prev_states:
            assert len(prev_s.get_next_states()) == 1
            for end_state in prev_s.get_next_states():
                score = prev_s.alpha
                if score > end_state.alpha:
                    end_state.alpha = score
                    end_state.bp = prev_s

        # backtrace
        best_tags = []
        bp_state = self.state_mat[-1][0]  # this is end state
        while True:
            bp_state = bp_state.bp
            if bp_state.bp is None:
                break
            best_tags.append(bp_state.curpos)
        best_tags.reverse()
        return best_tags

    # making feature vectores by label score and transition indicator
    # def build_feature_vector1(self, state):
    #     dim_vector = (NUMBER_OF_POS + 2) ** (self.left_n + self.right_n + 1) + 1
    #     assert dim_vector == self.model.n_feature
    #     vec = np.zeros((dim_vector, ), dtype=np.float64)
    #     # label score
    #     vec[0] = self.model.pos_score(t_step=state.t_step, pos_id=state.curpos)
    #     pos_seq = ' '.join([str(i) for i in state.pos_list])
    #     vec[self.pos_seq2id[pos_seq]] = 1.0
    #     state.set_feature_vector(vec)

    def build_feature_vector2(self, state):
        n_hopos_feature = HIDDEN + NUMBER_OF_POS  # dimention of features in hopos model
        dim_vector = NUMBER_OF_POS * n_hopos_feature
        assert dim_vector == self.model.n_feature
        vec = np.zeros((dim_vector, ), dtype=np.float64)
        curpos = state.curpos
        local_feature = self.model.hopos_feature(state.state_id)
        assert local_feature.shape[0] == n_hopos_feature
        vec[curpos*n_hopos_feature: (curpos+1)*n_hopos_feature] = local_feature
        state.set_feature_vector(vec)


class State(object):
    def __init__(self, pos_list, left_n, right_n, t_step):
        if pos_list != [INIT_STATE_LABEL] and pos_list != [END_STATE_LABEL]:
            self.leftpos = pos_list[:left_n]
            if right_n > 0:
                self.rightpos = pos_list[-right_n:]
            else:
                self.rightpos = []
            self.curpos = pos_list[left_n]
            assert len(self.leftpos + self.rightpos) + 1 == len(pos_list)
            self.alpha = float('-inf')
            self.feature_vector = None
        elif pos_list == [INIT_STATE_LABEL]:  # CAUTION !!
            self.alpha = 0
            self.phi = 1.
        else:
            self.alpha = float('-inf')
            self.phi = 1.
        self.score = None
        self.t_step = t_step
        self.fw_alpha = 1.0
        self.fw_beta = 1.0
        self.fw_log_alpha = 0.0
        self.fw_log_beta = 0.0
        self.sptag_score = None
        self.sptag = None
        self.next_states = []
        self.prev_states = []
        # self.label = pos_list
        self.pos_list = pos_list
        # self.alpha = float('-inf')  # CAUTION !!
        self.bp = None  # backpointer for viterbe algorithms
        # self.marginal_prob = None
        self.gold_flg = False  # whether this state is in gold path
        self.state_id = -1


    def add_prev_state(self, prev_s):
        self.prev_states.append(prev_s)

    def add_next_state(self, next_s):
        self.next_states.append(next_s)

    def get_cur_label(self):
        return self.curpos

    def get_label(self):
        return ' '.join(map(str, self.pos_list))

    def get_prev_states(self):
        return self.prev_states

    def get_next_states(self):
        return self.next_states

    def set_feature_vector(self, vec):
        self.feature_vector = vec

    def set_phi(self, phi):
        self.phi = phi


# judging whether we can connect two state nodes (from state1 to state2)
def connect_judge(state1, state2):
    l1, l2 = state1.get_label().split(), state2.get_label().split()
    return l1[1:] == l2[:-1]
