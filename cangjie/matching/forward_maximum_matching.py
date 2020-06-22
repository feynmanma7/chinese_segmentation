from cangjie.utils.dictionary import load_dictionary
import os


def forward_maximum_matching(word_dict=None,
                             max_num_char=None,
                             sentence=None):

    words = []
    start = 0

    while start < len(sentence):
        cur_word = None
        for len_word in range(max_num_char, 0, -1):
            maybe_word = sentence[start: start+len_word]
            if maybe_word in word_dict:
                cur_word = maybe_word
                start += len_word
                break

        if cur_word is None:
            cur_word = sentence[start]
            start += 1

        words.append(cur_word)

    return words


def test_seg_sentence(word_dict=None):
    test_sentence = "黑夜给了我黑色的眼睛，我却用它寻找光明。"
    # test_sentence = '南京市长江大桥'
    # test_sentence = u"这钟声，传递出中华民族从磨难中奋起的昂扬斗志，彰显出伟大民族精神在新时代焕发出的熠熠光辉。"

    # test_sentence = u"我是中华人民共和国的公民。"

    seg_results = forward_maximum_matching(word_dict=word_dict,
                                           max_num_char=5,
                                           sentence=test_sentence)

    print("Forward:", "/".join(seg_results))

    if "".join(seg_results) == test_sentence:
        print("Right Segmentation")
    else:
        print("Wrong Segmentation")


if __name__ == '__main__':
    data_dir = "/Users/flyingman/Developer/github/chinese_segmentation/data"
    dict_path = os.path.join(data_dir, "msr.dict")

    word_dict = load_dictionary(dict_path=dict_path)
    print("Total number of words is: %d\n" % (len(word_dict)))

    test_seg_sentence(word_dict=word_dict)