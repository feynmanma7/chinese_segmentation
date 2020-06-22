from cangjie.matching.forward_maximum_matching import forward_maximum_matching
from cangjie.matching.backward_maximum_matching import backward_maximum_matching


def count_single_char(words):
    cnt = 0
    for word in words:
        if len(word) == 1:
           cnt += 1
    return cnt


def bidirectional_maximum_matching(word_dict=None,
                                   max_num_char=None,
                                   sentence=None):

    forward_words = forward_maximum_matching(word_dict=word_dict,
                                             max_num_char=max_num_char,
                                             sentence=sentence)

    backward_words = backward_maximum_matching(word_dict=word_dict,
                                              max_num_char=max_num_char,
                                              sentence=sentence)

    """
    1. The less number of words, the better.
    2. If equal number of words, the less number of single character word, 
        the better.
    """

    if len(forward_words) < len(backward_words):
        return forward_words
    elif len(forward_words) > len(backward_words):
        return backward_words
    else:
        if count_single_char(forward_words) < count_single_char(backward_words):
            return forward_words
        else:
            return backward_words