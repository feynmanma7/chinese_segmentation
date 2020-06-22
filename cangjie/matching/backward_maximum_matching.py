from cangjie.utils.dictionary import load_dictionary
import os


def backward_maximum_matching(word_dict=None,
                              max_num_char=None,
                              sentence=None):

    words = []
    start = len(sentence)

    while start > 0:
        cur_word = None
        for len_word in range(max_num_char, 0, -1):
            maybe_word = sentence[max(0, start-len_word): start]
            if maybe_word in word_dict:
                cur_word = maybe_word
                start -= len_word
                break

        if cur_word is None:
            cur_word = sentence[start-1]
            start -= 1

        words.append(cur_word)

    words.reverse()
    return words