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

        return np.sum(alpha[steps - 1, :])


    def __update_alpha(self, hiddens, outputs):
        """
        alpha_0(s) = P(o_0, h_0=s | \\theta) = pi_s * emit_p(s, o_0)
        alpha_t(h_t=s) = { \\sum_{i = 0} ^ {S - 1} alpha_{t - 1}(i) * trans(i, s)} * emit_p(s, o_t)
        """
        h_0 = self.states[hiddens[0]]
        o_0 = self.vocabs[outputs[0]]
        self.alpha[0, h_0] = self.pi[h_0] * self.emit_p[h_0, o_0]

        for t in range(1, len(outputs)):
            h_t = self.states[hiddens[t]]
            o_t = self.vocabs[outputs[t]]

            self.alpha[t, h_t] = 0
            for i in range(self.n_s):
                self.alpha[t, h_t] += self.alpha[t-1, i] * self.trans_p[i, h_t]

            self.alpha[t, h_t] *=  self.emit_p[h_t, o_t]


    def __update_beta(self, hiddens, outputs):
        """
        beta_{T-1}(s) = 1,
        beta_t(s) = \sum_{h_{t+1}=j} trans_p(s, j) * emit_p(h_{t+1}=j, o_{t+1}) * beta_{t+1}(j)
        """
        h_T_sub = self.states[hiddens[-1]]
        self.beta[-1, h_T_sub] = 1
        for t in range(len(outputs)-2, -1, -1):
            h_t = self.states[hiddens[t]]
            o_t_plus = self.vocabs[outputs[t+1]]

            self.beta[t, h_t] = 0
            for j in range(self.n_s):
                self.beta[t, h_t] += self.trans_p[h_t, j] * self.emit_p[j, o_t_plus] * self.beta[t+1, j]


    def __update_gamma(self, hiddens, outputs):
        """
        gamma_t(s) = alpha_t(s) * beta_t(s) / { \\sum_{s'} alpha_t(s') *beta_t(s') }
        """
        for t in range(len(outputs)):
            total = 0

            for s in range(self.n_s):
                self.gamma[t, s] = self.alpha[t, s] * self.beta[t, s]
                total += self.gamma[t, s]

            for s in range(self.n_s):
                if total > 1e-6:
                    self.gamma[t, s] /= total


    def __update_pi(self, hiddens, outputs):
        # pi_s = gamma_0(s)
        h_0 = self.states[hiddens[0]]
        self.pi[h_0] = self.gamma[0, h_0]


    def __update_trans_p(self, hiddens, outputs):
        """
        trans_p(i, s) = \\sum_{t = 0} ^ {T - 2} \\xi_t{i, s} / \\sum_{t = 0} ^ {T - 2} \\gamma_t{i}
        """

        for i in range(self.n_s):
            for s in range(self.n_s):

                nominator = 0.
                denominator = 0.

                for t in range(0, len(outputs) - 1):
                    nominator += self.xi[t, i, s]
                    denominator += self.gamma[t, i]

                if denominator > 1e-6:
                    self.trans_p[i, s] = nominator / denominator


    def __update_emit_p(self, hiddens, outputs):
        """
        emit_p(s, v) = \\sum_{t = 0, o_t = v} ^ {t = T - 1} \\gamma_t(s) /
            \\sum_{t = 0} ^ {T - 1} \\gamma_t(s)
        """

        for step in range(len(outputs)):
            s = self.states[hiddens[step]]
            v = self.vocabs[outputs[step]]

            nominator = 0
            denominator = 0
            for t in range(len(outputs)):
                o_t = self.vocabs[outputs[t]]
                if o_t == v:
                    nominator += self.gamma[t, s]
                denominator += self.gamma[t, s]

            if denominator > 1e-6:
                self.emit_p[s, v] = nominator / denominator



    def __update_xi(self, hiddens, outputs):
        """
        \\xi_{T-1}(i, j) = P(h_{T-1}=i, h_{T}=j | O, \\theta) = 1 ?
        \\xi_t(i, j) = alpha_t(i) * trans_p{i, j} * emit_p_(j, o_{t+1}) * beta_{t + 1}(j) /
        { \\sum_{h_t=i}\\sum_{h_{t+1}=j} alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j) }
        """

        for i in range(self.n_s):
            for j in range(self.n_s):
                self.xi[-1, i, j] = 1

        for t in range(len(outputs)-1):
            #h_t = self.states[hiddens[t]]
            o_t_plus = self.vocabs[outputs[t+1]]

            total = 0
            for i in range(self.n_s):
                for j in range(self.n_s):
                    try:
                        self.xi[t, i, j] = self.alpha[t, i] * self.trans_p[i, j] * self.emit_p[t+1, o_t_plus] * self.beta[t+1, j]
                        total += self.xi[t, i, j]
                    except:
                        print(i, j, self.xi[t, i, j])
                        print(self.alpha[t, i])
                        #print(sel)

            for i in range(self.n_s):
                for j in range(self.n_s):
                    if total > 1e-6:
                        self.xi[t, i, j] /= total


    def statistic_train(self):
        pass

    def em_train(self):
        pass

    def train(self, train_generator, max_seq_len=None, method='statistic'):
        """
        <h1>Hidden Markov Models</h1>
        
        To estimate the parameters of HMM: [init_p, trans_p, emit_p]
        
        
        States = ['B', 'M', 'E', 'S']
        B: Begin of a word.
        M: Middle of a word.
        E: End of a word.
        S: Single characteristic word.
        
        
        Observed values: V = [vocab_0, vocab_1, ..., vocab_{N-1}], N = n_vocab
        Observed variables: O, Output sequences, o_t = v[o_t], t=0,1,..., T-1
        Hidden variables: H, The hidden sequences, which are the flag of segmentation,
            h_t, t=0,1, ..., T-1
        Complete data: (O, H),
        O = o_0, o_1, ..., o_{T-1}
        H = h_0, h_1, ..., h_{T-1}
        
        HMM assumptions:
        (1) Markov assumption, limited history, one-order Markov Chain,
            P(h_t | h_{t-1}, o_{t-1}, h_{t_2}, o_{t-2}..., h_0, o_0) = P(h_t | h_{t-1})
        (1) state is independent of time :  P(h_{t+1}=j | h_t=i) = P(h_{t'+1}=j | h_t'=i);
        (2) output only depends on current hidden state:
            P(o_t | h_{T-1}, o_{T-1}, ..., h_0, o_0} = P(o_t | h_t)
        
        
        Likelihood function of the Data:
        P(D | \\theta) = P(O | \\theta) = \\sum_h P(O, H | \\theta)
            = \\sum_h P(O | H; theta) * P(H | \\theta)
        
        Using Expectation-Maximization method,
        the Q-function is,
        Q(\\theta, \\theta^{old}) = E_H [ log P(O, H | \\theta) | O, \\theta^{old}]
        = \\sum_h log P(O, H | \\theta) * P(H | O, \\theta^{old})
        
        P(O, H | \\theta) =
            pi_{h_0} * emit_p(h_0, o_0)
            * trans_p(h_0, h_1) * emit_p(h_1, o_1)
            * trans_p(h_1, h_2) * emit_p(h_2, o_2)
            * ...
            * trans_p(h_{T-2}, h_{T-1}) * emit_p(h_{T-1}, o_{T-1})
        
        Thus,
        Q(\\theta, \\theta^{old}) = \\sum_h {
             log pi_{h_0} * P(H | O, \\theta^{old})                               # Only affects pi
            + \\sum_{i=0}^{T-2} log trans_p(h_i, h_{i+1}) * P(H | O, \\theta^{old})# Only affects trans_p
            + \\sum_{i=0}^{T-1} log emit_p(h_i, o_i) * P(H | O, \\theta^{old})     # Only affects emit_p
        }
        
        # === pi
        Considering components Q_pi only affects pi in Q, s.t. \\sum_s pi_s = 1
        Q_pi = \\sum_h log pi_{h_0} * P(H | \\theta^{old})
        = \\sum_{s=0}^{S-1} log pi_s * P(h_0 = s | O, \\theta^{old} )
        
        Let gamma_t(s) = P(h_t = s | O, \\theta)
        pi_s = gamma_0(s)
        
        Currently, I don't know why the forward and backward prob is defined, as below.
        
        Forward probability, given hmm model(\\theta), in time t, the hidden state is s,
        the observed sequence from time 0 to time t is o_0, o_1, ..., o_t,
        alpha_t(s) = P(o_0, o_1, ..., o_t, h_t = s | \\theta)
        alpha_0(s) = pi_s * trans_p(s, o_0)
        alpha_t(s) = {\\sum_{i=0}^{S-1} alpha_{t-1}(i) * trans(i, s)} * emit_p(s, o_t)
        P(O | \\theta) = \\sum_{s=0}^{S-1} alpha_{T-1}(s)
        
        
        Backward probability, given hmm model, in time t, the hidden state is s,
        the observed sequence from time t+1 to time T-1 is o_{t+1}, o_{t+2}, ..., o_{T-1}
        beta_t(s) = P(o_{t+1}, o_{t+2}, ..., o_{T-1} | h_t = s, \\theta)
        beta_t(s) = \sum_{h_{t+1}=j} P(h_{t+1=j}, o_{t+2}, ..., o_{T-1} | h_t=s, \\theta)
        = \sum_{j} P(o_{t+1}, o_{t+2}, ..., o_{T-1} | h_{t+1}=j, h_t=s, \\theta) # To use Markov assumption
          * P(h{t+1}=j | h_t=s, \\theta)  # trans_p(s, j)
        = \sum_{j} P(o_{t+2}, ..., o_{T-1} | o_{t+1}, h_{t+1}=j, h_t=s, \\theta)
          * P(o_{t+1} | h_{t+1}=j, h_t=s, \\theta) # emit_p(j, o_{t+1})
          * trans_p(s, j)
        = \sum_{j} beta_{t+1}(j) * emit_p(j, o_{t+1}) * trans_p(s, j)
        = \sum_{h_{t+1}=j} trans_p(s, j) * emit_p(h_{t+1}=j, o_{t+1}) * beta_{t+1}(j)
        
        
        gamma_t(s) = P(h_t = s | O, \\theta)
        = P(O, h_t = s | \\theta) / P(O | \\theta)
        
        P(O, h_t = s | \\theta) = P(o_0, o_1, ..., o_{T-1}, h_t = s | \\theta)
        = P(o_0, ..., o_t, h_t = s;  o_{t+1}, ..., o_{T-1} |  \\theta)
        = P(o_0, ..., o_t, h_t = s | \\theta)
                * P(o_{t+1}, ..., o_{T-1} | o_0, ..., o_t, h_t = s; \\theta)
        = alpha_t(s) * P(o_{t+1}, ..., o_{T-1} | h_t = s; \\theta)  # assumption of HMM
        = alpha_t(s) * beta_t(s)
        
        P(O | \\theta) = \\sum_{s=0}^{S-1} alpha_t(s) * beta_t(s)
        
        gamma_t(s) = alpha_t(s) * beta_t(s) /
            { \\sum_{s'} alpha_t(s') * beta_t(s') }
        
        
        Q_pi = \\sum_{s=0}^{S-1} log pi_s * gamma_0(s)
        s.t. \\sum_s pi_s = 1
        
        Use lagrange multiplier method,
        Lag(Q_pi, lambda) = Q_pi + lambda * (1 - \\sum_s pi_s)
        = \\sum_{h_0=0}^{h_0=S-1} log pi_s * gamma_0(s) + lambda * (1 - \\sum_s pi_s)
        s.t. lambda >= 0
        
        ==>
        lambda = \\sum_s gamma_0(s) = \\sum_s P(h_0 = s | O, \\theta) = 1
        pi_s = gamma_0(s) / lambda = gamma_0(s)
        
        # === trans_p
        Q(\\theta, \\theta^{old}) = \\sum_H {
             log pi_{h_0} * P(H | O, \\theta^{old})                               # Only affects pi
            + \\sum_{t=0}^{T-2} log trans_p(h_t, h_{t+1}) * P(H | O, \\theta^{old})# Only affects trans_p
            + \\sum_{t=0}^{T-1} log emit_p(h_t, o_t) * P(H | O, \\theta^{old})     # Only affects emit_p
        }
        
        Q_trans_p = \\sum_H \\sum_{t=0}^{T-2} {log trans_p(h_t, h_{t+1}) } * P(H | O, \\theta^{old})
        = \\sum_{i=0}^{S-1}
          \\sum_{j=0}^{S-1}
          \\sum_{t=0}^{T-2} {log trans_p(h_t=i, h_{t+1}=j)} * P(h_t=i, h_{t+1}=j | O, \\theta^{old})
          # HMM assumption,
            trans_p(h_t, h_{t+1}) = P(h_{t+1} | h_t)
            is independent of h_0, h_{t-1}, h_{t+1}, ..., h{T-1}
          # hence \\sum_H = \\sum_{h_0, h_1, ..., h_{T-1}} = \\sum_{h_t, h_{t+1}},
            P(H | O, \\theta) = P(h_t, h_{t+1} | O, \\theta)
        
        Let P(h_t=i, h_{t+1}=j | O, \\theta) = \\xi_t(i, j)
        
        Pre-compute for query (Dynamic Programming)
        \\xi_t(i, j) = P(h_t=i, h_{t+1}=j | O, \\theta)
        = P(h_t=i, h_{t+1}=j, O | \\theta) / P(O | \\theta)
        
        P(h_t=i, h_{t+1}=j, O | \\theta)
            = alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j)
        
        \\xi_t(i, j) = alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j) /
        { \\sum_{h_t=i}\\sum_{h_{t+1}=j} alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j) }
        
        
        Q_trans_p =
            \\sum_{i=0}^{S-1}
            \\sum_{j=0}^{S-1}
            \\sum_{t=0}^{T-2} {log trans_p(h_t=i, h_{t+1}=j)} * P(h_t=i, h_{t+1}=j | O, \\theta^{old})
        = \\sum_{h_t=i}\\sum_{h_{t+1}=j} {log trans_p(h_t, h_{t+1})} * \\xi_t{i, j}
        
        s.t. \\sum_s trans_p(i, s) = 1
        trans_p(i, s) = \\sum_{t=0}^{T-2} \\xi_t{i, s} / \\sum_{t=0}^{T-2} \\gamma_t{i}
        NOTE: from t=0 to T-2, total time steps is T-1.
        
        
        # === emit_p
        Q(\\theta, \\theta^{old}) = \\sum_H {
             log pi_{h_0} * P(H | O, \\theta^{old})                               # Only affects pi
            + \\sum_{t=0}^{T-2} log trans_p(h_t, h_{t+1}) * P(H | O, \\theta^{old})# Only affects trans_p
            + \\sum_{t=0}^{T-1} log emit_p(h_t, o_t) * P(H | O, \\theta^{old})     # Only affects emit_p
        }
        
        # HMM assumption, emit_p(h_t, o_t) = P(o_t | h_t),
            only depends on h_t, just need to enumerate on h_t
        Q_emit_p = \\sum_{s=0}^{S-1} \\sum_{t=0}^{T-1} {log emit_p(h_t, o_t)} * P(h_t=s | O, \\theta^{old})
        = \\sum_s \\sum_{t=0}^{T-1} {log emit_p(h_t, o_t)} * \\gamma_t(s)
        
        s.t. \\sum_{v=0}^{V-1} emit_{h_t, o_t=v} = 1
        emit_p(s, v) = \\sum_{t=0, o_t=v}^{t=T-1} \\gamma_t(s) / \\sum_{t=0}^{T-1} \\gamma_t(s)
        
        """
        """
        Summary,
        
        (1) init_p,
        pi_s = gamma_0(s),
        gamma_t(s) = alpha_t(s) * beta_t(s) /
        { \\sum_{s'} alpha_t(s') * beta_t(s') }
        
        alpha_0(s) = P(o_0, h_0=s | \\theta) = pi_s * emit_p(s, o_0)
        alpha_t(s) = { \\sum_{i=0}^{S-1} alpha_{t-1}(i) * trans(i, s) }   * emit_p(s, o_t)
        
        beta_{T-1}(s) = 1,
        beta_t(s) = \sum_{h_{t+1}=j} trans_p(s, j) * emit_p(h_{t+1}=j, o_{t+1}) * beta_{t+1}(j)
        
        (2) trans_p(i, s) = \\sum_{t=0}^{T-2} \\xi_t{i, s} / \\sum_{t=0}^{T-2} \\gamma_t{i}
        \\xi_t(i, j) = alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j) /
        { \\sum_{h_t=i}\\sum_{h_{t+1}=j} alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j) }
        
        (3) emit_p(s, v) = \\sum_{t=0, o_t=v}^{t=T-1} \\gamma_t(s) / \\sum_{t=0}^{T-1} \\gamma_t(s)
        """

        self.alpha = np.zeros((max_seq_len, self.n_s))
        self.beta = np.zeros((max_seq_len, self.n_s))
        self.gamma = np.zeros((max_seq_len, self.n_s))
        self.xi = np.zeros((max_seq_len, self.n_s, self.n_s))

        for outputs in train_generator:
            hiddens = []
            if len(outputs) == 1:
                hiddens = ['S']
            else:
                hiddens = ['M'] * len(outputs)
                hiddens[0] = 'B'
                hiddens[-1] = 'E'

            self.__update_alpha(hiddens, outputs)
            self.__update_beta(hiddens, outputs)
            self.__update_gamma(hiddens, outputs)
            self.__update_xi(hiddens, outputs)

            self.__update_pi(hiddens, outputs)
            self.__update_trans_p(hiddens, outputs)
            self.__update_emit_p(hiddens, outputs)

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

    def cut(self, sentence):
        """
        Let \\phi_t(j) be the maximum probability of hidden path h_0, h_1, ..., h_{t-1}, h_t

        \\phi_t(j) = max_{h_0, ..., h_{t-1}} P(h_t = j, h_{t-1}, ..., h_0, o_t, ..., o_0)

        \\phi_{t}(j) = max_{0<=i<=S-1} { \\phi_{t-1}(i) * trans_p(i, j) } * emit_p(j, o_t)

        \\phi_{0}(j) = pi(j) * emit_p(j, o_0)


        Let \\psi_t(j) be the `t-1`-th node of the hidden path h_0, h_1, ..., h_{t-1}, h_t
            with maximum prob.
        \\psi_t(j) = argmax_i \\phi_{t-1}(i) * trans_p(i, j)   # * emit_p(j, o_t)

        \\psi_0(j) = 0


        node_{T-1} = argmax_j \\phi_{T-1}(j)
        """

        T = len(sentence)

        phi = np.zeros((T, self.n_s))
        #psi = np.zeros((T, ))

        # \\phi_{0}(j) = pi(j) * emit_p(j, o_0)
        # \\phi_{t}(j) = max_{0<=i<=S-1} { \\phi_{t-1}(i) * trans_p(i, j) } * emit_p(j, o_t)

        o_0 = self.vocabs[sentence[0]]
        for j in range(self.n_s):
            phi[0, j] = self.pi[j] * self.emit_p[j, o_0]

        for t in range(1, len(sentence)):
            o_t = self.vocabs[sentence[t]]
            for j in range(self.n_s):
                phi[t, j] = 0
                for i in range(self.n_s):
                    if phi[t, j] < phi[t-1, i] * self.trans_p[i, j]:
                        phi[t, j] = phi[t-1, i] * self.trans_p[i, j]

                phi[t, j] *= self.emit_p[j, o_t]

        s = np.argmax(phi[-1, :])
        hiddens = [s]

        for t in range(len(sentence)-1, 0, -1):
            # \\psi_t(j) = argmax_i \\phi_{t-1}(i) * trans_p(i, j)

            j = hiddens[-1]
            prob = -1
            s = -1
            for i in range(self.n_s):
                if prob < phi[t-1, i] * self.trans_p[i, j]:
                    prob = phi[t-1, i] * self.trans_p[i, j]
                    s = i

            hiddens.append(s)

        print(hiddens[::-1])
        print(sentence)

        decode_states = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}

        print(''.join([decode_states[s] for s in hiddens]))

