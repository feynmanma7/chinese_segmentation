

def load_hmm():
    pass


if __name__ == '__main__':
    sentence = '4月29日，雄浑悠长的钟声响起，关闭了近百日的武汉黄鹤楼重新开门迎客。这钟声，传递出中华民族从磨难中奋起的昂扬斗志，彰显出伟大民族精神在新时代焕发出的熠熠光辉。'

    hmm = load_hmm()

    words = hmm.cut(sentence)
    print(words)