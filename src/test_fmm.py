from dict_utils import load_dict
from matching.forward_maximum_matching import fmm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dict", help="user_dict_path")
    args = parser.parse_args()

    if args.dict is not None:
        user_dict_path = args.dict
    else:
        user_dict_path = '../data/people.dict.pkl'

    print('user_dict_path', args.dict)

    sentence = '4月29日，雄浑悠长的钟声响起，关闭了近百日的武汉黄鹤楼重新开门迎客。这钟声，传递出中华民族从磨难中奋起的昂扬斗志，彰显出伟大民族精神在新时代焕发出的熠熠光辉。'

    user_dict = load_dict(dict_path=user_dict_path)
    print('Len user_dict=%d' % len(user_dict))

    # maximum characters of a word
    max_char = 5
    words = fmm(user_dict=user_dict, sentence=sentence, max_char=max_char)
    print('/ ' .join(words))