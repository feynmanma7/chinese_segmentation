from cangjie.utils.config import get_model_dir, get_data_dir
from cangjie.crf.rnn_crf import BiRNNCRF, test_model_once
from cangjie.utils.losses import mask_sparse_cross_entropy
from cangjie.utils.dictionary import load_dictionary
from cangjie.utils.preprocess import load_separator_dict
from cangjie.rnn.dataset import generate_seg_seq
import tensorflow as tf
tf.random.set_seed(7)
import numpy as np
import os, time


def model_predict(model=None, char_list=None, char2id_dict=None, separator_dict=None):
    # Test one sequence once.

    labels = []

    # If len(seq) < rnn_steps, add `pad`; Else truncate by rnn_steps.
    for char_seq_list, sub_label in generate_seg_seq(char_list=char_list,
                                                     separator_dict=separator_dict):

        if char_seq_list is None or len(char_seq_list) == 0:
            if sub_label is not None:
                labels.append(sub_label)
            continue

        char_index_list = []
        for char in char_seq_list:
            if char in char2id_dict:
                char_index_list.append(char2id_dict[char])
            else:
                char_index_list.append(1)  # 1 for `unk`

        char_index = np.array(char_index_list).reshape((1, len(char_index_list)))

        # [1, steps, 5]
        sub_probs = model(inputs=char_index)

        # [1, steps]
        sub_labels = tf.argmax(sub_probs, axis=2)

        # [steps, ]
        sub_labels = list(tf.squeeze(sub_labels, axis=0).numpy())

        if sub_label is not None:
            sub_labels.append(sub_label)

        labels.extend(sub_labels)

    return labels


def segmentation():
    vocab_size = 3954
    embedding_dim = 64
    rnn_units = 32
    pad_index = 0  # pad_index, to mask in loss
    rnn_steps = 30 # is not needed in test

    test_path = os.path.join(get_data_dir(), "msr_test.utf8")
    seg_path = os.path.join(get_data_dir(), "msr_test_birnn_crf.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")
    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict=%d" % len(char2id_dict))

    # === Build and compile model.
    model = BiRNNCRF(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    optimizer = tf.keras.optimizers.Adam(0.001)

    crf = model.crf_layer
    model.compile(optimizer=optimizer,
                  loss=crf.loss,
                  metrics=[crf.accuracy]
                  )

    """
    model.compile(optimizer=optimizer,
                   loss=mask_sparse_cross_entropy,
                   metrics=['acc'])
    """

    # === Load weights.
    checkpoint_dir = os.path.join(get_model_dir(), "rnn_model")
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    model.load_weights(checkpoint)

    # === Run once, to load weights of checkpoint.
    test_model_once(model=model, vocab_size=vocab_size)

    # Load separator_dict
    separator_dict = load_separator_dict()
    print("#separator_dict=%d" % len(separator_dict))

    fw = open(seg_path, 'w', encoding='utf-8')
    with open(test_path, 'r', encoding='utf-8') as f:

        line_cnt = 0
        for line in f:

            labels = model_predict(model=model,
                                   char_list=line[:-1],
                                   char2id_dict=char2id_dict,
                                   separator_dict=separator_dict)

            if len(line[:-1]) != len(labels):
                print("Wrong")
                print(line[:-1], '\n', labels)
                print(len(line[:-1]), len(labels))
                break

            # {0: pad, 1: B, 2: M, 3: E, 4: S}
            words = []
            word = []
            for i, label in zip(range(len(line)-1), labels):
                word.append(line[i])
                if label == 3 or label == 4:
                    words.append("".join(word))
                    word = []
            if len(word) > 0:
                words.append("".join(word))
            fw.write(" ".join(words) + '\n')

            line_cnt += 1
            if line_cnt % 100 == 0:
                print(line_cnt)

        print(line_cnt)
    fw.close()


if __name__ == '__main__':
    start = time.time()
    segmentation()
    end = time.time()
    last = end - start
    print("Segmentation done! Lasts %.2fs" % last)



