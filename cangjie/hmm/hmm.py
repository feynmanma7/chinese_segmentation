import numpy as np
np.random.seed(7)
import os
import pickle


class HMM():
    def __init__(self):
        super(HMM, self).__init__()

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

    def train(self, train_path=None, model_path=None):
        #states: {'B': 0, 'M': 1, 'E': 2, 'S': 3}

        # pi: [4], num_state
        self.pi = np.zeros((4))

        # trans_p: [4, 4], num_state * num_state
        self.trans_p = np.zeros((4, 4))

        # emit_p: [4, n_V] num_state * vocab_size
        self.emit_p = {0: {}, 1: {}, 2: {}, 3: {}}

        total_num_word = 0

        with open(train_path, 'r', encoding='utf-8') as f:

            line_cnt = 0

            for line in f.readlines():
                line_cnt += 1
                if line_cnt % 10000 == 0:
                    print(line_cnt)

                if len(line) == 1: #'\n'
                    continue


                pre_state = None

                for word in line[:-1].split(' '):
                    word = word.strip()
                    if len(word) == 0:
                        continue

                    if len(word) == 1:
                        hidden = [3]
                    else:
                        hidden = [1] * len(word)
                        hidden[0] = 0
                        hidden[-1] = 2

                    assert len(hidden) == len(word)

                    total_num_word += 1

                    # Accumulate pi
                    self.pi[hidden[0]] += 1

                    # Accumulate trans_p
                    for state in hidden:
                        if pre_state is not None:
                            self.trans_p[pre_state, state] += 1
                        pre_state = state

                    # Accumulate emit_p
                    for state, char in zip(hidden, word):
                        if char not in self.emit_p[state]:
                            self.emit_p[state][char] = 1
                        else:
                            self.emit_p[state][char] += 1

            print(line_cnt)

        assert total_num_word > 0

        # Normalize pi
        self.pi /= total_num_word

        # Normalize trans_p: [num_state, num_state]
        self.trans_p /= np.sum(self.trans_p, axis=1).reshape((4, 1))

        # Normalize emit_p: [num_state, num_char_of_this_state:{char: count}]
        for state in self.emit_p:
            total_num = sum(self.emit_p[state].values())
            assert total_num > 0

            for char in self.emit_p[state].keys():
                self.emit_p[state][char] /= total_num

        print(self.pi)
        print(self.trans_p)
        print(self.emit_p)

        self.save_model(model_path=model_path)

    def save_model(self, model_path=None):
        model = {'pi': self.pi, 'tran_p': self.trans_p, 'emit_p': self.emit_p}
        with open(model_path, 'wb') as fw:
            pickle.dump(model, fw)

        print('Save model done!', model_path)

    def load_model(self, model_path=None):

        with open(model_path, 'rb') as fr:
            model = pickle.load(fr)

            assert 'pi' in model
            self.pi = model['pi']

            assert 'trans_p' in model
            self.trans_p = model['trans_p']

            assert 'emit_p' in model
            self.emit_p = model['emit_p']

        print("Load model done!", model_path)


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

