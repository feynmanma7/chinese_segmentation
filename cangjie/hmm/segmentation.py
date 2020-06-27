from cangjie.hmm.hmm import HMM
from cangjie.utils.config import get_data_dir, get_model_dir
from cangjie.utils.dictionary import load_dictionary

import os


def seg_on_sentence(hmm, sentence,
                    is_use_matching=None,
                    matching_method=None,
                    max_num_char=None,
                    word_dict=None):
    seg_words = hmm.decode(sentence=sentence,
                           is_use_matching=is_use_matching,
                           matching_method=matching_method,
                           max_num_char=max_num_char,
                           word_dict=word_dict)

    return seg_words


def seg_on_file(model=None, test_path=None, test_result_path=None,
                is_use_matching=None,
                matching_method=None,
                max_num_char=None,
                word_dict=None):
    fw = open(test_result_path, 'w', encoding='utf-8')

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            seg_words = seg_on_sentence(model, line[:-1],
                                        is_use_matching=is_use_matching,
                                        matching_method=matching_method,
                                        max_num_char=max_num_char,
                                        word_dict=word_dict)
            if seg_words is None:
                fw.write("\n")
            else:
                fw.write(" ".join(seg_words) + "\n")

    fw.close()


if __name__ == "__main__":
    data_dir = get_data_dir()
    model_dir = get_model_dir()

    model_path = os.path.join(model_dir, "hmm", "hmm.pkl")
    test_path = os.path.join(data_dir, "msr_test.utf8")
    test_result_path = os.path.join(data_dir, "msr_test_hmm.utf8")
    dict_path = os.path.join(data_dir, "msr.dict")

    word_dict = load_dictionary(dict_path=dict_path)
    print("Total number of words is: %d\n" % (len(word_dict)))

    hmm = HMM()
    hmm.load_model(model_path=model_path, is_training=False)

    seg_res = seg_on_sentence(hmm, sentence='黑夜给了我黑色的眼睛，我却用它寻找光明。')
    print("/".join(seg_res))
    seg_on_file(model=hmm,
                test_path=test_path,
                test_result_path=test_result_path,
                is_use_matching=True,
                matching_method="bimm",
                max_num_char=6,
                word_dict=word_dict)

    print("Segmentation done!", test_result_path)

