from cangjie.max_entropy.maximum_entropy import MaximumEntropy
from cangjie.hmm.preprocess import load_vocab


def get_pre_char(arr, index):
    if index >= 0:
        return arr[index]
    return '-'


def get_next_char(arr, index):
    if index < len(arr):
        return arr[index]
    return '-'


def func_cur(x_char=None, x_idx=None, y=None, xy_dict=None):
    if (x_idx, y) in xy_dict:
        return 1

    return 0


def is_symbol(char):
    if char in {c: True for c in
                ['（', '）', '，', '。', '！',
                 '【', '】', '；', '？', '——',
                 ',', '.', '"', '、', '!',
                 '(', ')', '@', '?', '”']}:
        return True
    return False


def func_is_symbol(x_char=None, x_idx=None, y=None, xy_dict=None):
    if is_symbol(x_char) and y == 3: # 'S': 3
        return 1

    return 0


if __name__ == '__main__':
    vocab_path = '../../data/people_char_vocab.pkl'
    train_data_path = '../../data/people.txt'
    states = ['B', 'M', 'E', 'S']
    features = [func_cur, func_is_symbol]
    model_dir = '../../models/max_entropy'
    epochs = 1000

    vocab = load_vocab(vocab_path=vocab_path)
    #print(vocab)

    max_ent = MaximumEntropy(line_limit=100)
    max_ent.train(features=features,
                  states=states,
                  vocab=vocab,
                  train_path=train_data_path,
                  epochs=epochs)

    sentences = "我是中国人"
    seg_words = max_ent.predict(sentences=sentences)
    print(seg_words)
