from src.hmm.hmm import HMM
from src.hmm.preprocess import load_vocab

def mini_generator(corpus):
    for line in corpus:
        for words in line.split(' '):
            yield words


def mini_train():
    states = ['B', 'M', 'E', 'S']
    vocabs = {'我': 0, '是': 1, '中': 2, '国': 3, '人': 4, '家': 5}
    corpus = ['我 是 中国 人', '中国 是 我 家']

    hmm = HMM(vocabs=vocabs)
    #hmm.train(train_generator=mini_generator(corpus), max_seq_len=2)
    #hmm.save_model(model_dir='../../models/hmm')
    hmm.load_model(model_dir='../../models/hmm')

    hmm.cut(sentence='我是中国人')


def transform(words, vocabs=None):
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
        words_hiddens, words_outputs = transform(words.strip(), vocabs)
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


def train():
    vocab_path = '../../data/people_char_vocab.pkl'
    vocabs = load_vocab(vocab_path)
    train_data_path = '../../data/people.txt'

    gen = train_generator(train_data_path, vocabs=vocabs)
    states = ['B', 'M', 'E', 'S']
    hmm = HMM(vocabs=vocabs, states=states)
    hmm.train(train_generator=gen)
    model_dir = '../../models/hmm'
    hmm.save_model(model_dir=model_dir)

def test():
    vocab_path = '../../data/people_char_vocab.pkl'
    vocabs = load_vocab(vocab_path)
    train_data_path = '../../data/people.txt'

    gen = train_generator(train_data_path, vocabs=vocabs)
    states = ['B', 'M', 'E', 'S']
    hmm = HMM(vocabs=vocabs, states=states)
    #hmm.train(train_generator=gen)
    model_dir = '../../models/hmm'
    #hmm.save_model(model_dir=model_dir)
    hmm.load_model(model_dir=model_dir)

    sentence = "我是中国人，我爱我的祖国"
    decode_states={0: 'B', 1: 'M', 2: 'E', 3: 'S'}
    hiddens = hmm.decode(outputs=sentence, decode_states=decode_states)

    words = hmm.format_hiddens(hiddens, sentence)

    print(hiddens)
    print('/ '.join(words))

    sentence = '4月29日，雄浑悠长的钟声响起，关闭了近百日的武汉黄鹤楼重新开门迎客。这钟声，传递出中华民族从磨难中奋起的昂扬斗志，彰显出伟大民族精神在新时代焕发出的熠熠光辉。'
    hiddens = hmm.decode(outputs=sentence, decode_states=decode_states)
    words= hmm.format_hiddens(hiddens, sentence)
    print('/ '.join(words))



if __name__ == '__main__':
    #mini_train()
    #train()
    test()
