import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU


class Encoder(tf.keras.Model):
    def __init__(self,
                 vocab_size=10,
                 embedding_dim=16,
                 rnn_units=8
                 ):
        super(Encoder, self).__init__()

        self.embedding_layer = Embedding(input_dim=vocab_size+2,
                                         output_dim=embedding_dim) # 0 for `pad`, 1 for `unk`
        self.gru_layer = GRU(units=rnn_units,
                             return_sequences=False,
                             return_state=False)

    def call(self, inputs=None):
        # inputs: [None, steps]
        # [None, steps, embedding_dim]
        embedding = self.embedding_layer(inputs)

        # [None, rnn_units], return_sequence = False
        gru = self.gru_layer(embedding)

        return gru


class Decoder(tf.keras.Model):
    def __init__(self, num_states=4,
                 embedding_dim=16,
                 rnn_units=8):
        super(Decoder, self).__init__()

        # {0: `pad`, 1: 'B', 2: 'M', 3: 'E', 4: 'S', 5: '<start>'}
        self.embedding_layer = Embedding(input_dim=num_states+2,
                                         output_dim=embedding_dim)

        self.gru = GRU(units=rnn_units,
                       return_sequences=True,
                       return_state=True)

        # {0: `pad`, 1: 'B', 2: 'M', 3: 'E', 4: 'S'}
        self.softmax = Dense(units=num_states+1, activation='softmax')

    def call(self, targets=None, pre_hidden_state=None, training=None, mask=None):
        # targets: [None, steps]
        embedding = self.embedding_layer(targets)

        # gru: [None, steps, embedding_dim]
        # hidden_state: [None, embedding_dim]
        gru, hidden_state = self.gru(embedding, initial_state=pre_hidden_state)

        # [None, steps, num_states+1]
        softmax = self.softmax(gru)

        return softmax, hidden_state


def test_seq2seq_once(encoder=None, decoder=None, inputs=None, targets=None):
    encoder_hidden_state = encoder(inputs=inputs)
    print('encoder_hidden_state', encoder_hidden_state.shape)

    softmax, hidden_state = decoder(targets=targets, pre_hidden_state=encoder_hidden_state)
    print('softmax', softmax.shape, 'decoder_hidden_state', hidden_state.shape)


if __name__ == "__main__":
    batch_size = 2
    rnn_steps = 5
    vocab_size = 10
    num_states = 4
    embedding_dim = 16
    rnn_units = 8

    inputs = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=vocab_size+2, dtype=tf.int32)
    targets = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=num_states+1, dtype=tf.int32)

    encoder = Encoder(vocab_size=vocab_size,
                      embedding_dim=embedding_dim,
                      rnn_units=rnn_units)

    decoder = Decoder(num_states=num_states,
                      embedding_dim=embedding_dim,
                      rnn_units=rnn_units)

    test_seq2seq_once(encoder=encoder, decoder=decoder, inputs=inputs, targets=targets)