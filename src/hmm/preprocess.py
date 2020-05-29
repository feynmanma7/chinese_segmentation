import os
import pickle


def process_file(file_path, fw):
    with open(file_path, 'r', encoding='utf-8') as f:
        print(file_path)
        for line in f:
            buf = line.split(' ')
            fw.write(' '.join([pair.strip().split('/')[0].strip() for pair in buf]) + '\n')



def preprocess(input_dir, output_path):
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
    # Get split word_idx list
    corpus_dir = '../../data/2014'
    output_path = '../../data/people.txt'

    preprocess(corpus_dir, output_path)

    vocab_path = '../../data/people_char_vocab.pkl'
    generate_char_vocab(output_path, vocab_path)