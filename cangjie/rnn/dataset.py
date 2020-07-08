from cangjie.utils.config import get_data_dir
from cangjie.utils.preprocess import pad_sequence
from cangjie.rnn.dictionary import load_dictionary
import tensorflow as tf
import numpy as np
import os


def dataset_generator(data_path=None,
                      epochs=10,
                      shuffle_buffer_size=1024,
                      batch_size=16,
                      steps=10,
                      pad_index=0,
                      char2id_dict=None):
    # input_data: char_idx :: char_idx :: char_idx \t state_idx \s state_idx \s state_idx
    def generator():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                buf = line[:-1].split('\t')
                chars = buf[0].split('::')
                states = list(buf[1]) # labels are split by space.

                if len(chars) != len(states):
                    continue

                for i in range(len(chars) // steps + 1):
                    sub_chars = []
                    for char in chars[i*steps: min((i+1)*steps, len(chars))]:
                        if char in char2id_dict:
                            sub_chars.append(char2id_dict[char])
                        else:
                            sub_chars.append(1) # 1 for `unk`, 0 for `pad`
                    sub_states = states[i*steps: min((i+1)*steps, len(chars))]

                    inputs = sub_chars if len(sub_chars) == steps \
                        else pad_sequence(seq=sub_chars, max_len=steps, pad_index=pad_index)
                    outputs = sub_states if len(sub_states) == steps \
                        else pad_sequence(seq=sub_states, max_len=steps, pad_index=pad_index)

                    yield inputs, outputs

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=((steps, ), (steps, )),
                                             output_types=(tf.int32, tf.int32))

    return dataset.repeat(epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
        .batch(batch_size=batch_size)


def get_dataset(data_path=None,
                epochs=10,
                shuffle_buffer_size=1024,
                batch_size=16,
                steps=10,
                pad_index=0,
                char2id_dict=None):
    return dataset_generator(data_path=data_path,
                             epochs=epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             batch_size=batch_size,
                             steps=steps,
                             pad_index=pad_index,
                             char2id_dict=char2id_dict)


if __name__ == "__main__":
    train_path = os.path.join(get_data_dir(), "msr_training_label.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")

    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict = %d" % len(char2id_dict))

    train_dataset = get_dataset(data_path=train_path,
                                batch_size=4,
                                steps=10,
                                char2id_dict=char2id_dict)

    # inputs: [batch_size, steps]  \in [0, 1, 2, ..., vocab_size]
    # outputs: [batch_size, steps] \in [0, 1, 2, 3, 4]

    for i, (inputs, outputs) in zip(range(2), train_dataset):
        print(i, inputs.shape, outputs.shape)