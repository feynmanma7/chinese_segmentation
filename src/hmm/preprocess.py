import os
import pickle


def process_pair(pair):
    """
    Do the hand-dirty work to get clean word.
    1. for word like `[公共/b`, `[` is concat with real word.
    """
    if pair is None or len(pair) == 0:
        return None

    word = pair.strip().split('/')[0].strip()
    if len(word) == 0:
        return None

    if word[0] == '[':
        return word[1:]
    return word


def process_file(file_path, fw):
    with open(file_path, 'r', encoding='utf-8') as f:
        print(file_path)

        for line in f:
            words = []
            buf = line.split(' ')
            for pair in buf:
                word = process_pair(pair.strip())
                if word is not None:
                    words.append(word)

            if len(words) > 0:
                fw.write(' '.join(words) + '\n')


def get_split_words(input_dir, output_path):
    fw = open(output_path, 'w', encoding='utf-8')

    for dir_name in os.listdir(input_dir):
        dir_path = os.path.join(input_dir, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                process_file(file_path, fw)
            except:
                import traceback
                traceback.print_exc()

    fw.close()
    print("Preprocess done!", output_path)


def generate_char_vocab(input_path, vocab_path):
    vocab = {}
    idx = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            for char in line:
                if char in ['', ' ', '\n', '\t']:
                    continue
                if char not in vocab:
                    vocab[char] = idx
                    idx += 1

    with open(vocab_path, 'wb') as fw:
        pickle.dump(vocab, fw)
    print('Write char_vocab done!', vocab_path)


def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as fr:
        vocab = pickle.load(fr)
        return vocab


if __name__ == '__main__':
    corpus_dir = '../../data/2014'
    processed_path = '../../data/people.txt'
    char_vocab_path = '../../data/people_char_vocab.pkl'

    # Get split word list, in the format of {words words}, split by space ' '
    get_split_words(corpus_dir, processed_path)

    # get char_vocab, in the format of {char \t index(start from 0)}
    generate_char_vocab(processed_path, char_vocab_path)