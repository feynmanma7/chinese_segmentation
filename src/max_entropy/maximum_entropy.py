import numpy as np
import copy


class MaximumEntropy:
    def __init__(self, line_limit=None):
        super(MaximumEntropy, self).__init__()

        self.line_limit = line_limit # for debug

    def _compute_sigma(self, index=None):
        sigma = 1 / self.M * np.log()

    def _update_weights(self):
        for i in range(self.n_feature):
            self.weights[i] += self._compute_sigma(index=i)

    def _process_line(self, line):
        # Input: line: words words words, split by space
        # Output: Compute self.f_i, self.P_xy, self.P_x

        X_char = [] # Output sequences
        X_idx = [] # Output index list of chars
        Y = [] # hidden states, 0: 'B', 1: 'M', 2: 'E', 3: 'S'

        for word in line.strip().split(' '):
            if len(word) == 0:
                continue
            if len(word) == 1:
                y = [3] # 'S', Single
                x_idx = [self.vocab[word]]
            else:
                x_idx = [self.vocab[char] for char in word]
                y = [1] * len(word) # 'M', Middle
                y[0] = 0  # 'B', Begin
                y[-1] = 2 # 'E', End

            X_char += word
            X_idx += x_idx
            Y += y

        for (x_char, x_idx, y) in zip(X_char, X_idx, Y):
            for i in range(self.n_feature):

                if (x_idx, y) not in self.feature_dict[i]:
                    self.feature_dict[i][(x_idx, y)] = True

                f_i_xy = self.features[i](x_char=x_char, x_idx=x_idx, y=y, xy_dict=self.feature_dict[i])
                if f_i_xy == 1:
                    self.f[i][(x_idx, y)] = f_i_xy

            if (x_idx, y) not in self.P_xy:
                self.P_xy[(x_idx, y)] = 1
            else:
                self.P_xy[(x_idx, y)] += 1

            if x_idx not in self.P_x:
                self.P_x[x_idx] = 1
            else:
                self.P_x[x_idx] += 1

    def _init_params(self,
                     features=None,
                     states=None,
                     vocab=None,
                     input_path=None):

        """
        1. M, num_features
        2. f_i(x, y), i=1, 2, ..., num_features, [feauture_i, {(X=x, Y=y), f_i(x, y)=f_i}]
        3. P_(x, y)  {(X=x, Y=y)}
        4. P_(x) {X=x}
        5. P(y|x)   exp{\sum_i w_i * f_i(x, y)} / Z_w
        6. EP_(f_i(x, y))  \sum_{x, y} P_(x, y) * f_i(x, y)  [num_features, (x, y)]
        7. EP(f_i(x, y))  \sum_{x, y} P_(x) * P(y|x) * f_i(x, y) [num_features, (x, y)]
        """

        self.features = features
        self.n_feature = len(features)
        self.weights = np.zeros(self.n_feature)
        self.vocab = vocab
        self.M = self.n_feature # f^#(x, y) = \sum_{i=1}^{n_F} f_i(x, y), max (\sum_i f_i(x, y)), num_features

        n_state = len(states)
        self.n_state = n_state
        n_v = len(vocab)

        self.f = [{} for _ in range(self.n_feature)]
        # f[i] = {(x, y): 1}

        self.P_xy = {}
        self.P_x = {}
        self.EP_ = [0] * self.n_feature

        self.feature_dict = [{} for _ in range(self.n_feature)]
        # feature_dict[i]: {(x, y): cnt}

        with open(input_path, 'r', encoding='utf-8') as f:
            line_cnt = 0
            for line in f:
                self._process_line(line[:-1])
                line_cnt += 1

                if self.line_limit is not None and line_cnt > self.line_limit:
                    break

        P_xy_cnt = 0
        for x_idx, cnt in self.P_xy.items():
            P_xy_cnt += cnt
        if P_xy_cnt > 0:
            for (x, y), xy_cnt in self.P_xy.items():
                # {(x, y): xy_cnt} ==> {(x, y): P_(x, y)}
                self.P_xy[(x, y)] /= P_xy_cnt

        P_x_cnt = 0
        for x_idx, cnt in self.P_x.items():
            P_x_cnt += cnt
        if P_x_cnt > 0:
            for x, x_cnt in self.P_x.items():
                # {x: x_cnt} ==> {x: P_(x)}
                self.P_x[x] /= P_x_cnt

        # EP_i = \sum_{x, y} P_(x, y) f_i(x, y)
        # f_i: {(x, y): value of f_i_xy, 0 or 1 for binary feature}
        for i in range(self.n_feature):
            self.EP_[i] = 0
            for (x, y), f_i_xy in self.f[i].items():
                if (x, y) in self.P_xy:
                    p_xy = self.P_xy[(x, y)]
                    self.EP_[i] += p_xy * f_i_xy


    def _compute_PYX(self):
        """
        P(y|x) = exp{ \sum_i w_i * f_i(x, y) } / Z_w

        Z_w = \sum_y P(y|x)

        P(Y|X): [y, {x: p(y|x) unnormalized}]
        Z_w = \sum_y P(Y|X)
        P(Y|X) /= Z_w

        \sum_y P(y|X) = 1
        """

        # Raw: [n_Y, n_X]
        # Actually: [n_Y, {x: P(y|x)}] to save memory
        PYX = [{} for _ in range(self.n_state)]

        for i in range(self.n_feature):
            for (x, y), f_i_xy in self.f[i].items():
                if x not in PYX[y]:
                    PYX[y][x] = self.weights[i] * f_i_xy
                else:
                    PYX[y][x] += self.weights[i] * f_i_xy

        Z_w = [0] * self.n_state
        for y in range(self.n_state):
            for x, pyx in PYX[y].items():
                PYX[y][x] = np.exp(pyx)
                Z_w[y] += PYX[y][x]

        for y in range(self.n_state):
            for x, pyx in PYX[y].items():
                PYX[y][x] /= Z_w[y]

        return PYX

    def _compute_EP(self):
        """
        EPi = \sum_{x, y} P_(x) * P(y|x) * f_i(x, y)
        """
        EP = [0] * self.n_feature

        PYX = self._compute_PYX()

        for i in range(self.n_feature):
            EPi = 0

            for (x, y), f_i_xy in self.f[i].items():
                if x not in self.P_x or x not in PYX[y]:
                    continue

                p_x = self.P_x[x]
                pyx = PYX[y][x]
                EPi += p_x * pyx * f_i_xy

            EP[i] = EPi

        return EP

    def _is_convergence(self, old_weights, cur_weights, threshold=1e-3):
        diff = abs(np.sum(old_weights - cur_weights))

        if diff < threshold:
            return True

        return False


    def predict(self, words):
        pass

    def train(self, features=None,
              states=None,
              vocab=None,
              epochs=None,
              train_path=None):

        self._init_params(features=features,
                          states=states,
                          vocab=vocab,
                          input_path=train_path)

        for epoch in range(epochs):
            EP = self._compute_EP()
            old_weights = copy.deepcopy(self.weights)

            for i, w in enumerate(self.weights):
                self.weights[i] += 1.0 / self.M * np.log(self.EP_[i] / EP[i])

            diff = abs(np.sum((self.weights - old_weights)))
            print('Epoch: {:d}, weights diff: {:.4f}'.format(epoch, diff))
            print(self.weights)
            if diff < 1e-3:
                break

        print(self.weights)


