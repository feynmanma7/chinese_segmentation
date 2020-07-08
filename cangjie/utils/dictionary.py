import os
import pickle
from cangjie.utils.config import get_data_dir


def load_dictionary(dict_path=None):
    with open(dict_path, 'rb') as fr:
        my_dict = pickle.load(fr)
        return my_dict

    return None


def genearte_dictionary(train_path=None, dict_path=None):
    """
    train_data:  Labeled words split by space.
    word_dict:  {word: word_count}
    """

    # Word counting.
    word_dict = {}
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line[:-1].split(' '):
                word = word.strip()
                if len(word) == 0:
                    continue

                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1

    # Save word_dict to disk.
    with open(dict_path, 'wb') as fw:
        pickle.dump(word_dict, fw)


if __name__ == "__main__":
    data_dir = get_data_dir()
    train_path = os.path.join(data_dir, "msr_training.utf8")
    dict_path = os.path.join(data_dir, "msr.dict")

    genearte_dictionary(train_path=train_path, dict_path=dict_path)
    print("Generate word dictionary to %s done!" % dict_path)