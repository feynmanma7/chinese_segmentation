import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size=None):
        super(Encoder, self).__init__()
        self.input_dim = vocab_size + 1

    def call(self, inputs=None):
        # inputs: [batch_size, steps]
        # embedding: [batch_size, steps, embedding_dim]
        embedding = Embedding(input_dim=self.input_dim,
                              output_dim=32)(inputs)
        print(embedding.shape)

        # embedding: [batch_size, steps, embedding_dim]
        # rnn_1:[batch_size, steps, units_1]
        rnn_1 = GRU(units=32,
                       activation='sigmoid',
                       use_bias=True,
                       return_sequences=True,
                       stateful=False)(embedding)
        print(rnn_1.shape)

        # rnn_1: [batch_size, steps, units_1]
        # outputs: [batch_size, units_2]
        outputs, hidden_state = GRU(units=16,
                       activation='sigmoid',
                       use_bias=True,
                       return_sequences=False,
                       return_state=True,
                       stateful=False)(rnn_1)

        print(outputs.shape)
        print(hidden_state.shape)

        return hidden_state


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

    def call(self, encoder_hidden_state=None, targets=None):
        embedding = Embedding(input_dim=4, output_dim=16)(targets)

        # [batch_size, targets_steps, units]
        rnn_1 = GRU(units=16, activation='sigmoid', return_sequences=True)\
            (inputs=embedding, initial_state=encoder_hidden_state)
        print(rnn_1.shape)
        return rnn_1

if __name__ == "__main__":
    batch_size = 2
    steps = 7

    vocab_size = 10
    inputs = tf.random.uniform((batch_size, steps), minval=0, maxval=vocab_size, dtype=tf.int32)

    states = 4
    targets = tf.random.uniform((batch_size, steps), minval=0, maxval=states, dtype=tf.int32)

    print("encoder")
    encoder = Encoder(vocab_size=vocab_size)

    encoder_hidden_state = encoder(inputs=inputs)
    print(encoder_hidden_state.shape)

    print("decoder")
    decoder = Decoder()
    outputs = decoder(targets=targets, encoder_hidden_state=encoder_hidden_state)
    print(outputs.shape)


