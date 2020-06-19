from cangjie.hmm.hmm import HMM
from cangjie.hmm.preprocess import load_vocab
import numpy as np
import scipy.stats as st
#np.random.seed(7)


def sample_start(pi, ):
    init_begin = pi[0]
    init_single = pi[-3]

    prob_begin = init_begin / (init_begin + init_single)
    if np.random.random() < prob_begin:
        start_state = 0 # 'B', begin
    else:
        start_state = 3 # 'S', single

    return start_state


def binary_search(r, cdf):
    start = 1
    end = len(cdf) - 2
    while start <= end:
        mid = (start + end) // 2
        if r <= cdf[mid]:
            if cdf[mid-1] < r:
                return mid
            end = mid - 1
        else:
            start = mid + 1


def order_search(r, cdf):
    for i in range(1, len(cdf)-1):
        if cdf[i-1] < r and r <= cdf[i]:
            return i


def sample_output(state=None, cdfs=None):
    r = np.random.random()
    idx = binary_search(r, cdfs[state])
    return idx - 1 #


def compute_cdf(probs):
    cdf = [-1.]
    total = 0
    for prob in probs:
        total += prob
        cdf.append(total)

    cdf.append(2.)

    return cdf


def generate_text():
    vocab_path = '../../data/people_char_vocab.pkl'
    model_dir = '../../models/hmm'
    states = ['B', 'M', 'E', 'S']

    vocabs = load_vocab(vocab_path)
    query_vocabs = {idx:char for char, idx in vocabs.items()}
    hmm = HMM(vocabs=vocabs, states=states)
    hmm.load_model(model_dir=model_dir)

    pi = hmm.pi
    tran_p = hmm.trans_p # [S, S]
    emit_p = hmm.emit_p  # [S, V]

    # [S, S]
    trans_cdfs = [compute_cdf(tran_p[s, :]) for s in range(tran_p.shape[0])]

    # [S, V]
    emit_cdfs = [compute_cdf(emit_p[s, :]) for s in range(emit_p.shape[0])]

    state_idx = sample_start(pi)
    out_idx = sample_output(state_idx, emit_cdfs)
    out_char = query_vocabs[out_idx]

    num_text = 1000
    print(out_char, end='')

    for i in range(num_text-1):
        state_idx = sample_output(state=state_idx, cdfs=trans_cdfs)
        out_idx = sample_output(state=state_idx, cdfs=emit_cdfs)
        out_char = query_vocabs[out_idx]
        print(out_char, end='')
        if (i+1) % 50 == 0:
            print()


if __name__ == '__main__':
    generate_text()
