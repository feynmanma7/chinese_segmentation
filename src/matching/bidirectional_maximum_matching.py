from .forward_maximum_matching import fmm
from .backward_maximum_matching import bmm


def count_single_char(words):
    cnt = 0
    for word in words:
        if len(word) == 1:
           cnt += 1
    return cnt


def bi_mm(user_dict=None, sentence=None, max_char=None):
    forward_words = fmm(user_dict=user_dict,
                        sentence=sentence, max_char=max_char)
    backward_words = bmm(user_dict=user_dict,
                        sentence=sentence, max_char=max_char)

    bi_words = []

    """
    1. The less number of words, the better.
    2. If eqaul number of words, the less number of single character word, 
        the better.
    """
    if len(forward_words) < len(backward_words):
        print('less #words of fmm')
        bi_words = forward_words
    elif len(forward_words) > len(backward_words):
        print('less #words of bmm')
        bi_words = backward_words
    else:
        print('equal #words of fmm&bmm', end=', ')
        if count_single_char(forward_words) < count_single_char(backward_words):
            print('less #single_char of fmm')
            bi_words = forward_words
        else:
            print('less #single_char of bmm')
            bi_words = backward_words

    print('\nfmm:')
    print('/'.join(forward_words))
    print('\nbmm:')
    print('/'.join(backward_words))
    print('\nbi_mm:')
    print('/'.join(bi_words))

    return bi_words

