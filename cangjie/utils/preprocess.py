def pad_sequence(seq=None, max_len=None, pad_index=None):
    # seq: List
    assert len(seq) < max_len
    return seq + [pad_index] * (max_len - len(seq))