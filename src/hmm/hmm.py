class HMM():
    def __init__(self):
        """
        states: hidden states can't be observed. [n_state, 1]
            s = ['B', 'M', 'E', 'S']
        n_state: number of hidden states, 4
        n_out: number of output vocabulary size.

        hidden_states: H = [h_0, h_1, ..., h_{T-1}], h_t = ['B', 'M', 'E', 'S']

        trans_p: Transform probability matrix between hidden states. [n_state, n_state]
            trans_p[i, j] = P(h_t = s_j | h_{t-1} = s_i)

        output_sequences, O = [o_0, o_1, ..., o_{T-1}], o_t = [vocab_0, vocab_1, ..., vocab_{N-1}],
            N is the char_vocab_size

        emit_p: Emit probability matrix from hidden state to observed value. [n_state, n_out]
            emit_p[s, v] = P(o_t = v | h_t = s)

        init_p: Initial probability list of each hidden states. [n_state, 1]
            init_p[s] = P(h_0 = s)
        """
        pass

    def call(self):
        pass

    def cut(self, sentence):
        pass

    def train(self):
        """
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
        P(D | \theta) = P(O | \theta) = \sum_h P(O, H | \theta)
            = \sum_h P(O | H; theta) * P(H | \theta)

        Using Expectation-Maximization method,
        the Q-function is,
        Q(\theta, \theta^{old}) = E_H [ log P(O, H | \theta) | O, \theta^{old}]
        = \sum_h log P(O, H | \theta) * P(H | O, \theta^{old})

        P(O, H | \theta) =
            pi_{h_0} * emit_p(h_0, o_0)
            * trans_p(h_0, h_1) * emit_p(h_1, o_1)
            * trans_p(h_1, h_2) * emit_p(h_2, o_2)
            * ...
            * trans_p(h_{T-2}, h_{T-1}) * emit_p(h_{T-1}, o_{T-1})

        Thus,
        Q(\theta, \theta^{old}) = \sum_h {
             log pi_{h_0} * P(H | O, \theta^{old})                               # Only affects pi
            + \sum_{i=0}^{T-2} log trans_p(h_i, h_{i+1}) * P(H | O, \theta^{old})# Only affects trans_p
            + \sum_{i=0}^{T-1} log emit_p(h_i, o_i) * P(H | O, \theta^{old})     # Only affects emit_p
        }

        # === pi
        Considering components Q_pi only affects pi in Q, s.t. \sum_s pi_s = 1
        Q_pi = \sum_h log pi_{h_0} * P(H | \theta^{old})
        = \sum_{s=0}^{S-1} log pi_s * P(h_0 = s | O, \theta^{old} )

        Let gamma_t(s) = P(h_t = s | O, \theta)
        pi_s = gamma_0(s)

        Currently, I don't know why the forward and backward prob is defined, as below.

        Forward probability, given hmm model(\theta), in time t, the hidden state is s,
        the observed sequence from time 0 to time t is o_0, o_1, ..., o_t,
        alpha_t(s) = P(o_0, o_1, ..., o_t, h_t = s | \theta)
        alpha_0(s) = pi_s * trans_p(s, o_0)
        alpha_t(s) = {\sum_{i=0}^{S-1} alpha_{t-1}(i) * emit_p(i, s)} * trans_p(s, o_t)
        P(O | \theta) = \sum_{s=0}^{S-1} alpha_{T-1}(s)


        Backward probability, given hmm model, in time t, the hidden state is s,
        the observed sequence from time t+1 to time T-1 is o_{t+1}, o_{t+2}, ..., o_{T-1}
        beta_t(s) = P(o_{t+1}, o_{t+2}, ..., o_{T-1} | h_t = s, \theta)

        gamma_t(s) = P(h_t = s | O, \theta)
        = P(O, h_t = s | \theta) / P(O | \theta)

        P(O, h_t = s | \theta) = P(o_0, o_1, ..., o_{T-1}, h_t = s | \theta)
        = P(o_0, ..., o_t, h_t = s;  o_{t+1}, ..., o_{T-1} |  \theta)
        = P(o_0, ..., o_t, h_t = s | \theta)
                * P(o_{t+1}, ..., o_{T-1} | o_0, ..., o_t, h_t = s; \theta)
        = alpha_t(s) * P(o_{t+1}, ..., o_{T-1} | h_t = s; \theta)  # assumption of HMM
        = alpha_t(s) * beta_t(s)

        P(O | \theta) = \sum_{s=0}^{S-1} alpha_t(s) * beta_t(s)

        gamma_t(s) = alpha_t(s) * beta_t(s) /
            { \sum_{s'} alpha_t(s') * beta_t(s') }


        Q_pi = \sum_{s=0}^{S-1} log pi_s * gamma_0(s)
        s.t. \sum_s pi_s = 1

        Use lagrange multiplier method,
        Lag(Q_pi, lambda) = Q_pi + lambda * (1 - \sum_s pi_s)
        = \sum_{h_0=0}^{h_0=S-1} log pi_s * gamma_0(s) + lambda * (1 - \sum_s pi_s)
        s.t. lambda >= 0

        ==>
        lambda = \sum_s gamma_0(s) = \sum_s P(h_0 = s | O, \theta) = 1
        pi_s = gamma_0(s) / lambda = gamma_0(s)

        # === trans_p
        Q(\theta, \theta^{old}) = \sum_H {
             log pi_{h_0} * P(H | O, \theta^{old})                               # Only affects pi
            + \sum_{t=0}^{T-2} log trans_p(h_t, h_{t+1}) * P(H | O, \theta^{old})# Only affects trans_p
            + \sum_{t=0}^{T-1} log emit_p(h_t, o_t) * P(H | O, \theta^{old})     # Only affects emit_p
        }

        Q_trans_p = \sum_H \sum_{t=0}^{T-2} {log trans_p(h_t, h_{t+1}) } * P(H | O, \theta^{old})
        = \sum_{i=0}^{S-1}
          \sum_{j=0}^{S-1}
          \sum_{t=0}^{T-2} {log trans_p(h_t=i, h_{t+1}=j)} * P(h_t=i, h_{t+1}=j | O, \theta^{old})
          # HMM assumption,
            trans_p(h_t, h_{t+1}) = P(h_{t+1} | h_t)
            is independent of h_0, h_{t-1}, h_{t+1}, ..., h{T-1}
          # hence \sum_H = \sum_{h_0, h_1, ..., h_{T-1}} = \sum_{h_t, h_{t+1}},
            P(H | O, \theta) = P(h_t, h_{t+1} | O, \theta)

        Let P(h_t=i, h_{t+1}=j | O, \theta) = \xi_t(i, j)

        Pre-compute for query (Dynamic Programming)
        \xi_t(i, j) = P(h_t=i, h_{t+1}=j | O, \theta)
        = P(h_t=i, h_{t+1}=j, O | \theta) / P(O | \theta)

        P(h_t=i, h_{t+1}=j, O | \theta)
            = alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j)

        \xi_t(i, j) = alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j) /
        { \sum_{h_t=i}\sum_{h_{t+1}=j} alpha_t(i) * trans_p{i, j} * emit_p_{t+1}(j) * beta_{t+1}(j) }


        Q_trans_p =
            \sum_{i=0}^{S-1}
            \sum_{j=0}^{S-1}
            \sum_{t=0}^{T-2} {log trans_p(h_t=i, h_{t+1}=j)} * P(h_t=i, h_{t+1}=j | O, \theta^{old})
        = \sum_{h_t=i}\sum_{h_{t+1}=j} {log trans_p(h_t, h_{t+1})} * \xi_t{i, j}




        # === emit_p
        Q(\theta, \theta^{old}) = \sum_H {
             log pi_{h_0} * P(H | O, \theta^{old})                               # Only affects pi
            + \sum_{t=0}^{T-2} log trans_p(h_t, h_{t+1}) * P(H | O, \theta^{old})# Only affects trans_p
            + \sum_{t=0}^{T-1} log emit_p(h_t, o_t) * P(H | O, \theta^{old})     # Only affects emit_p
        }

        # HMM assumption, emit_p(h_t, o_t) = P(o_t | h_t),
            only depends on h_t, just need to enumerate on h_t
        Q_emit_p = \sum_{s=0}^{S-1} \sum_{t=0}^{T-1} {log emit_p(h_t, o_t)} * P(h_t=s | O, \theta^{old})


        """