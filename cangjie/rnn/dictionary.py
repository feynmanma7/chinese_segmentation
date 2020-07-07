from cangjie.utils.config import get_data_dir
import pickle, os


def load_dictionary(dict_path=None):
    with open(dict_path, 'rb') as fr:
        my_dict = pickle.load(fr)

    return my_dict


def build_dictionary(input_path=None, char2id_dict_path=None):
    # input_data:  word '\s' word

    char2id_dict = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        # index = 0, left for padding.
        index = 1
        for line in f:
            buf = line[:-1].split(' ')
            if len(buf) == 0:
                continue
            for word in buf:
                for char in word:
                    if char not in char2id_dict:
                        char2id_dict[char] = index
                        index += 1

    with open(char2id_dict_path, 'wb') as fw:
        pickle.dump(char2id_dict, fw)

    print("Len #char2id_dict = %d" % len(char2id_dict))
    return True


if __name__ == '__main__':
    train_path = os.path.join(get_data_dir(), "msr_training.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_rnn_dict.pkl")

    build_dictionary(input_path=train_path, char2id_dict_path=char2id_dict_path)