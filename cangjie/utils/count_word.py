from cangjie.utils.config import get_data_dir
import os, pickle


def count_word(input_path=None, word_cnt_dict_path=None):
    """
    input_data: training data of segmentation, split by space.
    word_cnt_dict: {word: count}
    """
    word_cnt_dict = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            buf = line[:-1].split(' ')
            for word in buf:
                if word not in word_cnt_dict:
                    word_cnt_dict[word] = 1
                else:
                    word_cnt_dict[word] += 1

    with open(word_cnt_dict_path, 'wb') as fw:
        pickle.dump(word_cnt_dict, fw)


if __name__ == '__main__':
    train_path = os.path.join(get_data_dir(), "msr_training.utf8")
    word_cnt_dict_path = os.path.join(get_data_dir(), "msr_training_word_cnt_dict.pkl")

    count_word(input_path=train_path, word_cnt_dict_path=word_cnt_dict_path)
    print('Write done!', word_cnt_dict_path)