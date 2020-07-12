from cangjie.utils.dictionary import load_dictionary
from cangjie.matching.forward_maximum_matching import forward_maximum_matching
from cangjie.matching.backward_maximum_matching import backward_maximum_matching
from cangjie.matching.bidirectional_maximum_matching import bidirectional_maximum_matching
from cangjie.utils.config import get_data_dir
import os


def seg_on_sentence(sentence=None, word_dict=None, method=None, max_num_char=None):
    if method == "bmm":
        seg_func = backward_maximum_matching
    elif method == "fmm":
        seg_func = forward_maximum_matching
    elif method == "bimm":
        seg_func = bidirectional_maximum_matching
    else:
        return False

    seg_words = seg_func(
        word_dict=word_dict,
        max_num_char=max_num_char,
        sentence=sentence)

    return seg_words


def seg_on_file(test_path=None, seg_path=None, word_dict=None, method=None, max_num_char=None):
    fw = open(seg_path, 'w', encoding='utf-8')

    line_cnt = 0
    with open(test_path, 'r', encoding='utf-8') as fr:
        if method == "bmm":
            seg_func = backward_maximum_matching
        elif method == "fmm":
            seg_func = forward_maximum_matching
        elif method == "bimm":
            seg_func = bidirectional_maximum_matching
        else:
            return False

        for line in fr:
            line_seg_result = seg_func(
                word_dict=word_dict,
                max_num_char=max_num_char,
                sentence=line[:-1])

            fw.write(" ".join(line_seg_result) + '\n')

            line_cnt += 1
            if line_cnt % 100 == 0:
                print(line_cnt)

    fw.close()
    print(line_cnt)

    return True


if __name__ == '__main__':
    data_dir = get_data_dir()
    dict_path = os.path.join(data_dir, "msr.dict")
    test_path = os.path.join(data_dir, "msr_test.utf8")
    method = "bimm"
    max_num_char = 6

    test_result_path = os.path.join(data_dir, "msr_test_" + method + ".utf8")

    word_dict = load_dictionary(dict_path=dict_path)
    print("Total number of words is: %d\n" % (len(word_dict)))

    seg_on_file(word_dict=word_dict,
                test_path=test_path,
                seg_path=test_result_path,
                method=method,
                max_num_char=max_num_char)

    print(test_result_path)