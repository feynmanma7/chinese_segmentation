from cangjie.utils.config import get_model_dir, get_data_dir
from cangjie.rnn.rnn import RNNSeg, test_rnnseg_once
from cangjie.utils.losses import mask_sparse_cross_entropy
from cangjie.utils.dictionary import load_dictionary
from cangjie.utils.preprocess import load_separator_dict
import tensorflow as tf
import numpy as np
import os, time


def generate_seq(char_list=None, rnn_steps=None, separator_dict=None):
    seq = []
    for char in char_list:
        if char in separator_dict:
            sub_label = 4 # 4 for `S`ingle
            yield seq, sub_label
            seq = []
        else:
            if len(seq) == rnn_steps:
                yield seq, None
                seq = []
            else:
                seq.append(char)

    yield seq, None


def segmentation():
    vocab_size = 3954
    embedding_dim = 64
    rnn_units = 32
    pad_index = 0  # pad_index, to mask in loss
    rnn_steps = 30

    test_path = os.path.join(get_data_dir(), "msr_test.utf8")
    seg_path = os.path.join(get_data_dir(), "msr_test_rnn.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")
    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict=%d" % len(char2id_dict))

    # === Build and compile model.
    rnnseg = RNNSeg(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    optimizer = tf.keras.optimizers.Adam(0.001)

    rnnseg.compile(optimizer=optimizer,
                   loss=mask_sparse_cross_entropy,
                   metrics=['acc'])

    checkpoint_dir = os.path.join(get_model_dir(), "rnn_model")
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    rnnseg.load_weights(checkpoint)

    # === Run once, to load weights of checkpoint.
    test_rnnseg_once(rnnseg=rnnseg, vocab_size=vocab_size)

    #print(rnnseg.embedding_layer.get_weights())

    # Load separator_dict
    separator_dict = load_separator_dict()
    print("#separator_dict=%d" % len(separator_dict))

    fw = open(seg_path, 'w', encoding='utf-8')
    with open(test_path, 'r', encoding='utf-8') as f:

        for line in f:
            buf = line[:-1]
            labels = []

            # Test one sequence once.
            # If len(seq) < rnn_steps, add `pad`; Else truncate by rnn_steps.
            for char_seq_list, sub_label in generate_seq(char_list=buf,
                                         rnn_steps=rnn_steps,
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
                        char_index_list.append(1) # 1 for `unk`

                if len(char_index_list) < rnn_steps:
                    char_index_list += [0] * (rnn_steps - len(char_index_list))

                char_index = np.array(char_index_list).reshape((1, rnn_steps))
                #print(char_index.shape)

                # [1, steps, 5]
                sub_probs = rnnseg.predict(char_index)

                # [1, steps]
                sub_labels = tf.argmax(sub_probs, axis=2)
                sub_labels = list(tf.reshape(sub_labels, shape=(rnn_steps, )).numpy())[:len(char_seq_list)]

                if sub_label is not None:
                    sub_labels.append(sub_label)

                labels.extend(sub_labels)

            if len(buf) != len(labels):
                print(buf, labels)
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

    fw.close()


if __name__ == '__main__':
    start = time.time()
    segmentation()
    end = time.time()
    last = end - start
    print("Segmentation done! Lasts %.2fs" % last)



