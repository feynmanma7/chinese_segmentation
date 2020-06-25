import numpy as np
import pickle


class HMM():
    def __init__(self):
        super(HMM, self).__init__()

    def _get_emit_y_to_x(self, x=None):
        emit_y_to_x = np.zeros(4)
        for y in range(4):
            if x in self.emit_p[y]:
                emit_y_to_x[y] = self.emit_p[y][x]

        return emit_y_to_x

    def _viterbi(self, sentence=None):
        N = len(sentence)
        sigma = np.zeros((N, 4)) #4: num_states

        # === Start condition for sigma
        emit_y_to_x = self._get_emit_y_to_x(x=sentence[0])

        # self.pi: [4, ]
        # emit_y_to_x: [4, ]
        sigma[0, :] = self.pi * emit_y_to_x

        # === Recursion for sigma
        for t in range(1, N):
            emit_y_to_x = self._get_emit_y_to_x(x=sentence[t])

            # sigma: N * S
            # sigma_{t-1}: [S, ]
            # trans_p: [S * S]
            # emit_y_to_x: [S, ]
            for j in range(4):
                trans = np.dot(sigma[t - 1, :].T, self.trans_p)
                i = np.argmax(trans)
                sigma[t, j] = trans[i] * emit_y_to_x[j]

        # === Start for optimal node
        opt_y_list = []
        y = np.argmax(sigma[-1, :])

        opt_y_list.append(int(y))

        for t in range(N-2, -1, -1):
            # sigma[t, :]: [S, ]
            # self.trans_p: [S, y]
            y = np.argmax(sigma[t, :] * self.trans_p[:, y])
            opt_y_list.append(int(y))

        return opt_y_list[::-1]

    def decode(self, sentence=None):
        if len(sentence) == 0:
            return None
        elif len(sentence) == 1:
            return [sentence]
        else:
            y_list = self._viterbi(sentence=sentence)
            print(y_list)
            seg_res = self.format_hiddens(hiddens=y_list, outputs=sentence)
            print(seg_res)
            return seg_res

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

        self.save_model(model_path=model_path)

    def save_model(self, model_path=None):
        model = {'pi': self.pi, 'trans_p': self.trans_p, 'emit_p': self.emit_p}
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

    def format_hiddens(self, hiddens=None, outputs=None):
        assert len(hiddens) == len(outputs)

        words = []
        for i, (state, char) in enumerate(zip(hiddens, outputs)):
            words.append(char)
            if state == 2 or state == 3:
                if i != len(hiddens) - 1:
                    words.append('/')

        return ''.join(words).split('/')

