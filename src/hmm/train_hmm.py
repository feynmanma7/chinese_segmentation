from src.hmm.hmm import HMM
from src.hmm.preprocess import load_vocab


def transform_words(words, vocabs=None):
    # input:   words (char)
    # output:  hiddens, output_indexes
    #   hiddens: 0, 1, 2, 3
    #   output_indexes:  [idxOf(char) for char in words]

    # 'B': 0, 'M': 1, 'E': 2, 'S': 3
    if len(words) == 1:
        outputs = vocabs[words]
        return [3], [outputs]

    hiddens = [1] * len(words)  # 'M'
    hiddens[0] = 0  #'B'
    hiddens[-1] = 2 #'E'

    outputs = [vocabs[char] for char in words]

    return hiddens, outputs


def process_line(line, vocabs=None):
    hiddens, outputs = [], []
    for words in line.strip().split(' '):
        if len(words.strip()) == 0:
            continue
        words_hiddens, words_outputs = transform_words(words.strip(), vocabs)
        hiddens += words_hiddens
        outputs += words_outputs

    return hiddens, outputs


def train_generator(train_data_path, vocabs=None):
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                yield process_line(line, vocabs=vocabs)
            except:
                import traceback
                traceback.print_exc()


def train_hmm():
    vocab_path = '../../data/people_char_vocab.pkl'
    train_data_path = '../../data/people.txt'
    states = ['B', 'M', 'E', 'S']
    model_dir = '../../models/hmm'

    vocabs = load_vocab(vocab_path)
    gen = train_generator(train_data_path, vocabs=vocabs)

    hmm = HMM(vocabs=vocabs, states=states)
    hmm.train(train_generator=gen)

    hmm.save_model(model_dir=model_dir)


if __name__ == '__main__':
    train_hmm()

