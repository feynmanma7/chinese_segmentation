def fmm(user_dict=None, sentence=None, max_char=None):

    words = []
    start = 0

    while start < len(sentence):
        cur_word = None
        for len_word in range(max_char, 0, -1):
            maybe_word = sentence[start:start+len_word]
            if maybe_word in user_dict:
                cur_word = maybe_word
                start += len_word
                break

        if cur_word is None:
            cur_word = sentence[start]
            start += 1

        words.append(cur_word)

    return words

