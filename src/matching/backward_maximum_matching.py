def bmm(user_dict=None, sentence=None, max_char=None):

    words = []
    start = len(sentence)

    while start > 0:
        cur_word = None
        for len_word in range(max_char, 0, -1):
            maybe_word = sentence[max(0, start-len_word):start]
            if maybe_word in user_dict:
                cur_word = maybe_word
                start -= len_word
                break

        if cur_word is None:
            cur_word = sentence[start]
            start -= 1

        words.append(cur_word)

    words.reverse()
    return words
