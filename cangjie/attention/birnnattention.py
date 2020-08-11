import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, GRU
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
import numpy as np


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 dense_units=8,
                 embedding_dim=16):
        super(Attention, self).__init__()
        self.dense_units = dense_units
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.Q_layer = Dense(units=self.dense_units)
        self.K_layer = Dense(units=self.dense_units)
        self.V_layer = Dense(units=self.dense_units)

    def call(self, inputs, **kwargs):
        # inputs: [None, seq_len, rnn_units]
        # Attention(Q, K, V) = softmax(QK^T/\sqrt(d_k)) V

        # [None, seq_len, dense_units]
        Q = self.Q_layer(inputs)

        K = self.K_layer(inputs)
        V = self.V_layer(inputs)

        # [None, seq_len, seq_len]
        sim = tf.matmul(Q, K, transpose_b=True) / np.sqrt(self.embedding_dim)

        # [None, seq_len, seq_len]
        softmax = tf.nn.softmax(sim)

        # [None, seq_len, rnn_units]
        outputs = tf.matmul(softmax, V)

        return outputs


class BiRNNAttention(tf.keras.Model):
    def __init__(self, vocab_size=10,
                 num_states=4,
                 embedding_dim=16,
                 rnn_units=8):
        super(BiRNNAttention, self).__init__()

        # 0 for `pad`, 1 for `unk`
        self.embedding_layer = Embedding(input_dim=vocab_size+2,
                                         output_dim=embedding_dim)

        # merge_mode: sum, mul, concat, ave. Default is `concat`.
        self.bi_rnn_layer = Bidirectional(GRU(units=rnn_units, return_sequences=True),
                                           merge_mode="ave")

        self.attention_layer = Attention(dense_units=rnn_units, embedding_dim=embedding_dim)

        # 4-tag: B, M, E, S, `0` for pad
        self.dense_layer = Dense(units=5, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # inputs: [None, steps]

        # embedding: [None, steps, embedding_dim]
        embedding = self.embedding_layer(inputs)
        #print('embedding', embedding.shape)

        # rnn: [None, steps, rnn_units]
        rnn = self.bi_rnn_layer(embedding)
        #print('rnn', rnn.shape)

        attention = self.attention_layer(rnn)
        #print('attention', attention.shape)

        # softmax: [None, steps, 5]
        softmax = self.dense_layer(attention)

        return softmax


def test_model_once(model=None, vocab_size=10):
    batch_size = 2
    steps = 30

    # 0 for `pad`, 1 for `unk`
    inputs = tf.random.uniform((batch_size, steps), minval=0, maxval=vocab_size+2)
    print('inputs', inputs.shape)

    outputs = model(inputs)
    print('outputs', outputs.shape)

    return outputs


if __name__ == '__main__':
    vocab_size = 10
    model = BiRNNAttention(vocab_size=vocab_size)
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss=tf.losses.SparseCategoricalCrossentropy)
    test_model_once(model=model)


