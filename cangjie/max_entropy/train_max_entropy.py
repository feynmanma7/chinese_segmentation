from cangjie.max_entropy.maximum_entropy import MaximumEntropy
from cangjie.hmm.preprocess import load_vocab
import os


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
    base_dir = "/Users/flyingman/Developer/github/chinese_segmentation"

    instance_path = os.path.join(base_dir, "data/people_instance.txt")
    feature_path = os.path.join(base_dir, "data/people_feature.txt")
    model_dir = os.path.join(base_dir, "models/max_entropy")

    states = ['B', 'M', 'E', 'S']


    vocab_path = '../../data/people_char_vocab.pkl'
    train_data_path = '../../data/people.txt'

    features = [func_cur, func_is_symbol]
    model_dir = '../../models/max_entropy'
    epochs = 1000

    vocab = load_vocab(vocab_path=vocab_path)
    #print(vocab)

    max_ent = MaximumEntropy(line_limit=100, states=states)
    max_ent.train(feature_path=feature_path,
                  epochs=epochs)

    """
    sentences = "我是中国人"
    seg_words = max_ent.predict(sentences=sentences)
    print(seg_words)
    """
