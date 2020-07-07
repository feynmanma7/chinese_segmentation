from cangjie.utils.config import get_data_dir
from cangjie.rnn.dictionary import load_dictionary
import os


def get_label(input_path=None, label_path=None, char2id_dict=None):
    # input_data: word \s word \s
    # label:  char,char,char  \t label_index""label_index""label_index
    # States = {'pad': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}

    fw = open(label_path, 'w', encoding='utf-8')

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            buf = line[:-1].split(' ')
            if len(buf) == 0:
                continue

            chars = []
            labels = []

            for word in buf:
                if len(word) == 0:
                    continue
                elif len(word) == 1:
                    label = ['4']
                else:
                    label = ['2'] * len(word)
                    label[0] = '1'
                    label[-1] = '3'

                chars.extend(word)
                labels.extend(label)

            assert len(chars) == len(labels)
            fw.write("::".join(chars) + '\t' + "".join(labels) + '\n')

        fw.close()
        print("Write Done!", label_path)


if __name__ == '__main__':
    data_path = os.path.join(get_data_dir(), "msr_training.utf8")
    label_path = os.path.join(get_data_dir(), "msr_training_label.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_rnn_dict.pkl")

    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict = %d" % len(char2id_dict))

    get_label(input_path=data_path, label_path=label_path, char2id_dict=char2id_dict)