import numpy as np
np.random.seed(7)
import os
import pickle


class HMM():
    def __init__(self, states=None,
                 vocabs=None,
                 pi=None,
                 trans_p=None,
                 emit_p=None):
        super(HMM, self).__init__()

        self.states = states
        self.vocabs = vocabs

        self.S = len(self.states)
        self.V = len(self.vocabs)

        if pi is not None:
            self.pi = pi
        else:
            self.pi = np.random.random((self.S)) # init_prob

        if trans_p is not None:
            self.trans_p = trans_p
        else:
            self.trans_p = np.random.random((self.S, self.S))

        if emit_p is not None:
            self.emit_p = emit_p
        else:
            self.emit_p = np.random.random((self.S, self.V))

    def forward_evaluate(self, outputs):
        steps = len(outputs)
        S = len(self.states)
        alpha = np.zeros((steps, S))

        o_0 = self.vocabs[outputs[0]]
        alpha[0, :] = self.pi[:] * self.emit_p[:, o_0]

        for t in range(1, len(outputs)):
            o_t = self.vocabs[outputs[t]]
            for s in range(S):
                alpha[t, s] = np.sum(alpha[t - 1, :] * self.trans_p[:, s]) * self.emit_p[s, o_t]

        forward_prob = np.sum(alpha[steps - 1, :])
        return forward_prob

    def backward_evaluate(self, outputs):
        steps = len(outputs)
        S = len(self.states)
        beta = np.zeros((steps, S))

        beta[-1, :] = 1

        for t in range(len(outputs) - 2, -1, -1):
            o_t_plus = self.vocabs[outputs[t+1]]
            for i in range(S):
                beta[t, i] = np.sum(beta[t+1, :] * self.emit_p[:, o_t_plus] * self.trans_p[i, :])

        o_0 = self.vocabs[outputs[0]]
        backward_prob = np.sum(self.pi[:] * self.emit_p[:, o_0] * beta[0, :])
        return backward_prob

    def decode(self, outputs, decode_states=None):
        steps = len(outputs)
        S = len(self.states)

        phi = np.zeros((steps, S))
        o_0 = self.vocabs[outputs[0]]

        phi[0, :] = self.emit_p[:, o_0] * self.pi[:]

        for t in range(1, len(outputs)):
            o_t = self.vocabs[outputs[t]]
            for j in range(S):
                phi[t, j] = max(phi[t-1, :] * self.trans_p[:, j]) * self.emit_p[j, o_t]

        o_T = self.vocabs[outputs[-1]]
        h_T = np.argmax(phi[-1, :] * self.emit_p[:, o_T])
        hiddens = [h_T]
        h_t_plus_opm = h_T

        for t in range(len(outputs)-2, -1, -1):
            o_t = self.vocabs[outputs[t]]
            h_t = np.argmax(phi[t, :] * self.emit_p[:, o_t] * self.trans_p[:, h_t_plus_opm])
            hiddens.append(h_t)
            h_t_plus_opm = h_t

        hiddens = [decode_states[s] for s in hiddens[::-1]]
        return hiddens

    def train(self, train_generator):
        V = len(self.vocabs)
        S = len(self.states)

        self.pi = np.ones((S)) * 1e-6
        self.trans_p = np.ones((S, S)) * 1e-6
        self.emit_p = np.ones((S, V)) * 1e-6

        pi_count = {}
        for s in range(S):
            pi_count[s] = 0

        trans_count = {} # (S, S)
        for i in range(S):
            s_trans_count = {}
            for j in range(S):
                s_trans_count[j] = 0
            trans_count[i] = s_trans_count

        emit_count = {} # (S, V)
        for s in range(S):
            s_emit_count = {}
            for v in range(V):
                s_emit_count[v] = 0
            emit_count[s] = s_emit_count

        for hiddens, outputs in train_generator:
            for state in hiddens:
                pi_count[state] += 1

            for t in range(len(hiddens)-1):
                state_i = hiddens[t]
                state_j = hiddens[t+1]
                trans_count[state_i][state_j] += 1

            for t in range(len(hiddens)):
                s = hiddens[t]
                v = outputs[t]
                emit_count[s][v] += 1

        # pi
        total_hidden = 0
        for state, count in pi_count.items():
            total_hidden += count
        for s in range(S):
            self.pi[s] = pi_count[s] / total_hidden

        # T
        for state_i, state_i_count in trans_count.items():
            state_i_total = 0
            for state_j, state_j_count in state_i_count.items():
                state_i_total += state_j_count
                self.trans_p[state_i, state_j] = state_j_count
            self.trans_p[state_i, :] /= state_i_total


        # E
        for state, state_count in emit_count.items():
            total = 0
            for v, v_count in state_count.items():
                total += v_count
                self.emit_p[state, v] = v_count
            if total > 0:
                self.emit_p[state, :] /= total

        return

    def save_model(self, model_dir=None):
        pi_path = os.path.join(model_dir, 'pi.pkl')
        with open(pi_path, 'wb') as fw:
            pickle.dump(self.pi, fw)

        trans_p_path = os.path.join(model_dir, 'trans_p.pkl')
        with open(trans_p_path, 'wb') as fw:
            pickle.dump(self.trans_p, fw)

        emit_p_path = os.path.join(model_dir, 'emit_p.pkl')
        with open(emit_p_path, 'wb') as fw:
            pickle.dump(self.emit_p, fw)

        print('Save model done!', model_dir)

    def load_model(self, model_dir=None):
        pi_path = os.path.join(model_dir, 'pi.pkl')
        with open(pi_path, 'rb') as fr:
            self.pi = pickle.load(fr)

        trans_p_path = os.path.join(model_dir, 'trans_p.pkl')
        with open(trans_p_path, 'rb') as fr:
            self.trans_p = pickle.load(fr)

        emit_p_path = os.path.join(model_dir, 'emit_p.pkl')
        with open(emit_p_path, 'rb') as fr:
            self.emit_p = pickle.load(fr)

        print("Load model done!", model_dir)


    def format_hiddens(self, hiddens, outputs):
        if len(hiddens) != len(outputs):
            print("Not equal number of hiddens and outputs !")
            return ""

        words = []
        cur = []
        for state, char in zip(hiddens, outputs):
            if state == 'S':
                words += cur
                words += char
                cur = []
            elif state == 'B':
                cur += char
            elif state == 'E':
                cur += char
                words += [''.join(cur)]
                cur = []
            else:
                cur += char

        if len(cur) > 0:
            words += cur

        return words

