from cangjie.rnn.dictionary import load_dictionary # same with rnn
from cangjie.rnn.dataset import generate_seg_seq

from cangjie.utils.config import get_model_dir, get_data_dir
from cangjie.seq2seq.seq2seq import Encoder, Decoder, Seq2Seq, test_seq2seq_once
from cangjie.utils.preprocess import load_separator_dict
import os, time
import tensorflow as tf
import numpy as np


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
        # print(char_index.shape)

        # [1, steps]
        sub_labels = model(inputs=char_index)

        # [steps, ]
        sub_labels = list(tf.squeeze(sub_labels, axis=0).numpy())

        if sub_label is not None:
            sub_labels.append(sub_label)

        labels.extend(sub_labels)

    return labels


def segmentation():
    vocab_size = 3954
    embedding_dim = 64
    num_states = 4
    rnn_units = 32
    pad_index = 0  # pad_index, to mask in loss
    rnn_steps = 30

    test_path = os.path.join(get_data_dir(), "msr_test.utf8")
    seg_path = os.path.join(get_data_dir(), "msr_test_seq2seq.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")
    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict=%d" % len(char2id_dict))

    # === Model
    encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    decoder = Decoder(num_states=num_states, embedding_dim=embedding_dim, rnn_units=rnn_units)
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder)

    # === Optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)

    # === Checkpoint
    checkpoint_dir = os.path.join(get_model_dir(), "seq2seq")
    #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    status = checkpoint.restore(latest)
    status.assert_existing_objects_matched()

    # === Test once
    batch_size = 2
    inputs = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=vocab_size+2, dtype=tf.int32)
    targets = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=num_states+1, dtype=tf.int32)
    test_seq2seq_once(encoder=encoder, decoder=decoder, inputs=inputs, targets=targets)

    # === Test

    # Load separator_dict
    separator_dict = load_separator_dict()
    print("#separator_dict=%d" % len(separator_dict))

    fw = open(seg_path, 'w', encoding='utf-8')

    with open(test_path, 'r', encoding='utf-8') as f:

        line_cnt = 0
        for line in f:
            buf = line[:-1]

            labels = model_predict(model=seq2seq,
                                   char_list=buf,
                                   char2id_dict=char2id_dict,
                                   separator_dict=separator_dict)

            if len(buf) != len(labels):
                print("Wrong")
                print(buf, '\n', labels)
                print(len(buf), len(labels))
                break

            # {0: pad, 1: B, 2: M, 3: E, 4: S}
            words = []
            word = []
            for i, label in zip(range(len(buf)), labels):
                word.append(buf[i])
                if label == 3 or label == 4:
                    words.append("".join(word))
                    word = []
            if len(word) > 0:
                words.append("".join(word))
            fw.write(" ".join(words) + '\n')

            line_cnt += 1
            if line_cnt % 100 == 0:
                print(line_cnt)

    fw.close()


if __name__ == '__main__':
    start = time.time()
    segmentation()
    end = time.time()
    last = end - start
    print("Segmentation done! Lasts %.2fs" % last)
