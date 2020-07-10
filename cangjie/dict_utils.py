import os
from utils.time_util import print_lasts_time
import pickle


def load_dict(dict_path=None):
    with open(dict_path, 'rb') as frb:
        user_dict = pickle.load(frb)
    return user_dict


def process_file(file_path):
    word_cnt_dict = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            buf = line.split(' ')
            for word_speech in buf:
                word = word_speech.split('/')[0]
                if word not in word_cnt_dict:
                    word_cnt_dict[word] = 1
                else:
                    word_cnt_dict[word] += 1

    return word_cnt_dict


@print_lasts_time(units='s')
def generate_dict(corpus_dir=None, dict_path=None):

    people_word_cnt_dict = {}

    for dir_name in os.listdir(corpus_dir):
        dir_path = os.path.join(corpus_dir, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            print(file_path)

            try:
                word_cnt_dict = process_file(file_path)

                for word, cnt in word_cnt_dict.items():
                    if word not in people_word_cnt_dict:
                        people_word_cnt_dict[word] = cnt
                    else:
                        people_word_cnt_dict[word] += cnt

            except Exception as e:
                print('Exception', file_path)
                import traceback
                traceback.print_exc()

    with open(dict_path, 'wb') as fwb:
        pickle.dump(people_word_cnt_dict, fwb)
    return len(people_word_cnt_dict)


if __name__ == '__main__':
    # Un-compressed the people2014.tar.gz, get the dir of `2014` in `data`.
    corpus_dir = '../data/2014'
    dict_path = '../data/people.dict.pkl'

    num_words = generate_dict(corpus_dir=corpus_dir, dict_path=dict_path)
    print("Write dict done! Length of dict is %d" % num_words)


