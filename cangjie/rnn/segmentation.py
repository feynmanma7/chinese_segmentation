from cangjie.utils.config import get_model_dir, get_data_dir
from cangjie.rnn.rnn import RNNSeg, test_rnnseg_once
from cangjie.utils.losses import mask_sparse_cross_entropy
import tensorflow as tf
import os


def segmentation():
    vocab_size = 5168
    embedding_dim = 64
    rnn_units = 32
    pad_index = 0  # pad_index, to mask in loss

    rnnseg = RNNSeg(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    optimizer = tf.keras.optimizers.Adam(0.001)

    rnnseg.compile(optimizer=optimizer,
                   loss=mask_sparse_cross_entropy,
                   metrics=['acc'])

    checkpoint_dir = os.path.join(get_model_dir(), "rnn_model")
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    rnnseg.load_weights(checkpoint)

    test_rnnseg_once(rnnseg=rnnseg, vocab_size=vocab_size)

    print(rnnseg.embedding_layer.get_weights())


if __name__ == '__main__':
    segmentation()



