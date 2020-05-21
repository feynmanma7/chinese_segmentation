import sys
sys.path.append('../')
from dict_utils import load_dict


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


if __name__ == '__main__':
    user_dict_path = '../../data/people.dict.pkl'
    sentence = '4月29日，雄浑悠长的钟声响起，关闭了近百日的武汉黄鹤楼重新开门迎客。这钟声，传递出中华民族从磨难中奋起的昂扬斗志，彰显出伟大民族精神在新时代焕发出的熠熠光辉。！'

    user_dict = load_dict(dict_path=user_dict_path)
    print('Len user_dict=%d' % len(user_dict))

    # maximum characters of a word
    max_char = 5
    words = bmm(user_dict=user_dict, sentence=sentence, max_char=max_char)
    print('/ ' .join(words))