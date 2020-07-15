def pad_sequence(seq=None, max_len=None, pad_index=None):
    # seq: List
    assert len(seq) < max_len
    return seq + [pad_index] * (max_len - len(seq))


def load_separator_dict():
    import string
    from zhon import hanzi

    sep_dict = {}

    for char in string.punctuation:
        sep_dict[char] = True

    for char in hanzi.punctuation:
        sep_dict[char] = True

    seps = [' ', '\t', '\n']
    for char in seps:
        sep_dict[char] = True

    return sep_dict


