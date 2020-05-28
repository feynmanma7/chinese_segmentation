import numpy as np
from src.hmm.hmm import HMM


if __name__ == '__main__':
    T = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    E = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])

    #states = [0, 1, 2]
    states = {'a': 0, 'b': 1, 'c': 2}
    vocabs = {'red': 0, 'white': 1}

    hmm = HMM(states=states,
              vocabs=vocabs,
              pi=pi,
              trans_p=T,
              emit_p=E)

    O = ['red', 'white', 'red']

    f_prob = hmm.forward_evaluate(O)
    print(f_prob)

    b_prob = hmm.backward_evaluate(O)
    print(b_prob)

