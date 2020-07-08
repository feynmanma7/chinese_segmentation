from cangjie.utils.config import get_data_dir
import pickle, os


def load_dictionary(dict_path=None):
    with open(dict_path, 'rb') as fr:
        my_dict = pickle.load(fr)

    return my_dict


def build_char2id_dict(input_path=None, char2id_dict_path=None, min_char_count=5):
    # input_data:  word '\s' word
    # char_cnt_dict: {char: count}
    # output: char2id_dict: {char: index}, index start from 2
    #   index = 0, for pad.
    #   index = 1, for char_count <= min_char_count

    # === char count
    char_cnt_dict = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            buf = line[:-1].split(' ')
            if len(buf) == 0:
                continue
            for word in buf:
                for char in word:
                    if char not in char_cnt_dict:
                        char_cnt_dict[char] = 1
                    else:
                        char_cnt_dict[char] += 1

    # === char to index
    char2id_dict = {}
    index = 2
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            buf = line[:-1].split(' ')
            if len(buf) == 0:
                continue
            for word in buf:
                for char in word:
                    if char in char2id_dict:
                        continue

                    if char in char_cnt_dict \
                            and char_cnt_dict[char] > min_char_count:
                        char2id_dict[char] = index
                        index += 1
                    else:
                        char2id_dict[char] = 1 # `unk` for low frequent char

    with open(char2id_dict_path, 'wb') as fw:
        pickle.dump(char2id_dict, fw)

    print("#char_cnt_dict=%d, #char2id_dict.values()=%d" % (
        len(char_cnt_dict), len(set(char2id_dict.values()))))
    return True


if __name__ == '__main__':
    train_path = os.path.join(get_data_dir(), "msr_training.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")

    build_char2id_dict(input_path=train_path,
                       char2id_dict_path=char2id_dict_path,
                       min_char_count=5)