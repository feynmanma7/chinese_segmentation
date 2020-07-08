import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM


class RNNSeg(tf.keras.Model):
    def __init__(self, vocab_size=10,
                 num_states=4,
                 embedding_dim=16,
                 rnn_units=8):
        super(RNNSeg, self).__init__()

        # 0 for `pad`, 1 for `unk`
        self.embedding_layer = Embedding(input_dim=vocab_size+2,
                                         output_dim=embedding_dim)

        # merge_mode: sum, mul, concat, ave. Default is `concat`.
        self.bi_lstm_layer = Bidirectional(LSTM(units=rnn_units, return_sequences=True),
                                           merge_mode="ave")
        # 4-tag: B, M, E, S, `0` for pad
        self.dense_layer = Dense(units=5, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # inputs: [None, steps]

        # embedding: [None, steps, embedding_dim]
        embedding = self.embedding_layer(inputs)

        # rnn: [None, steps, rnn_units]
        rnn = self.bi_lstm_layer(embedding)

        # softmax: [None, steps, 4]
        softmax = self.dense_layer(rnn)

        return softmax


def test_rnnseg_once(rnnseg=None, vocab_size=10):
    batch_size = 2
    steps = 5

    # 0 for `pad`, 1 for `unk`
    inputs = tf.random.uniform((batch_size, steps), minval=0, maxval=vocab_size+2)
    """
    import numpy as np
    inputs = np.random.randint(low=0, high=vocab_size+2, size=steps*batch_size)\
        .reshape((batch_size, steps))
    """
    print('inputs', inputs.shape)

    softmax = rnnseg(inputs)
    print('softmax', softmax.shape)

    return inputs


if __name__ == '__main__':
    vocab_size = 10
    rnnseg = RNNSeg(vocab_size=vocab_size)
    test_rnnseg_once(rnnseg=rnnseg)

