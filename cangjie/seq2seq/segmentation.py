from cangjie.rnn.dictionary import load_dictionary # same with rnn

from cangjie.utils.config import get_model_dir, get_data_dir
from cangjie.seq2seq.seq2seq import Encoder, Decoder, test_seq2seq_once
import os
import tensorflow as tf


def segmentation():
    vocab_size = 3954
    embedding_dim = 64
    num_states = 4
    rnn_units = 32
    pad_index = 0  # pad_index, to mask in loss
    rnn_steps = 30

    test_path = os.path.join(get_data_dir(), "msr_test.utf8")
    seg_path = os.path.join(get_data_dir(), "msr_test_rnn.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")
    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict=%d" % len(char2id_dict))

    # === Model
    encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    decoder = Decoder(num_states=num_states, embedding_dim=embedding_dim, rnn_units=rnn_units)

    # === Optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)

    # === Checkpoint
    checkpoint_dir = os.path.join(get_model_dir(), "seq2seq")
    #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    status = checkpoint.restore(latest)
    status.status.assert_existing_objects_matched()

    # === Test once
    batch_size = 2
    inputs = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=vocab_size + 2, dtype=tf.int32)
    targets = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=num_states + 1, dtype=tf.int32)
    test_seq2seq_once(encoder=encoder, decoder=decoder, inputs=inputs, targets=targets)



if __name__ == '__main__':
    segmentation()