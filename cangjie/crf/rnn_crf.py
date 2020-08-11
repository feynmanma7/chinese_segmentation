import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, GRU
from tensorflow.keras.utils import plot_model
from tf2crf import CRF


class BiRNNCRF(tf.keras.Model):
    def __init__(self, vocab_size=10,
                 num_states=4,
                 embedding_dim=16,
                 rnn_units=8):
        super(BiRNNCRF, self).__init__()

        # 0 for `pad`, 1 for `unk`
        self.embedding_layer = Embedding(input_dim=vocab_size+2,
                                         output_dim=embedding_dim)

        # merge_mode: sum, mul, concat, ave. Default is `concat`.
        self.bi_rnn_layer = Bidirectional(GRU(units=rnn_units, return_sequences=True),
                                           merge_mode="ave")
        # 4-tag: B, M, E, S, `0` for pad
        self.dense_layer = Dense(units=5, activation='softmax')

        self.crf_layer = CRF()

    def call(self, inputs, training=None, mask=None):
        # inputs: [None, steps]

        # embedding: [None, steps, embedding_dim]
        embedding = self.embedding_layer(inputs)

        # rnn: [None, steps, rnn_units]
        rnn = self.bi_rnn_layer(embedding)

        # softmax: [None, steps, 4]
        softmax = self.dense_layer(rnn)

        outputs = self.crf_layer(softmax)

        return outputs


def test_model_once(model=None, vocab_size=10):
    batch_size = 2
    steps = 100

    # 0 for `pad`, 1 for `unk`
    inputs = tf.random.uniform((batch_size, steps), minval=0, maxval=vocab_size+2)
    """
    import numpy as np
    inputs = np.random.randint(low=0, high=vocab_size+2, size=steps*batch_size)\
        .reshape((batch_size, steps))
    """
    print('inputs', inputs.shape)

    outputs = model(inputs)
    print('outputs', outputs.shape)

    return outputs


if __name__ == '__main__':
    vocab_size = 10
    model = BiRNNCRF(vocab_size=vocab_size)
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss=tf.losses.SparseCategoricalCrossentropy)
    test_model_once(model=model)


