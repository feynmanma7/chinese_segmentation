from src.hmm.hmm import HMM
from src.hmm.preprocess import load_vocab


def test_hmm():
    vocab_path = '../../data/people_char_vocab.pkl'
    model_dir = '../../models/hmm'
    states = ['B', 'M', 'E', 'S']
    decode_states = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}

    vocabs = load_vocab(vocab_path)
    hmm = HMM(vocabs=vocabs, states=states)
    hmm.load_model(model_dir=model_dir)
    sentence = "我是中国人，我爱我的祖国"

    hiddens = hmm.decode(outputs=sentence, decode_states=decode_states)
    words = hmm.format_hiddens(hiddens, sentence)

    print(hiddens)
    print('/ '.join(words))

    sentence = '4月29日，雄浑悠长的钟声响起，关闭了近百日的武汉黄鹤楼重新开门迎客。' \
               '这钟声，传递出中华民族从磨难中奋起的昂扬斗志，彰显出伟大民族精神在新时代焕发出的熠熠光辉。'
    hiddens = hmm.decode(outputs=sentence, decode_states=decode_states)
    words= hmm.format_hiddens(hiddens, sentence)
    print('/ '.join(words))


if __name__ == '__main__':
    test_hmm()