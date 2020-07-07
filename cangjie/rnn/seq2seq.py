import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Dense


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size=10,
                 embedding_dim=32,
                 rnn_units=16):
        super(Encoder, self).__init__()

        self.input_dim = vocab_size + 1 # Plus `1` for <pad>, pad_idx=0
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units

        self.embedding_layer = Embedding(input_dim=self.input_dim,
                              output_dim=self.embedding_dim)
        self.gru_layer = GRU(units=self.rnn_units,
                       activation='sigmoid',
                       use_bias=True,
                       return_sequences=False,
                       return_state=False,
                       stateful=False)

    def call(self, inputs=None):
        # inputs: [None, steps]
        # embedding: [None, steps, embedding_dim]
        embedding = self.embedding_layer(inputs)

        # hidden_state: [None, rnn_units]
        hidden_state = self.gru_layer(embedding)

        return hidden_state


class Decoder(tf.keras.Model):
    def __init__(self, num_states=4, embedding_dim=32, rnn_units=16):
        super(Decoder, self).__init__()

        self.num_states = num_states
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units

        # pad_idx = 0
        self.embedding_layer = Embedding(input_dim=self.num_states+1,
                                         output_dim=self.embedding_dim)
        self.gru_layer = GRU(units=self.rnn_units,
                             activation='sigmoid',
                             return_sequences=False,
                             return_state=True)
        self.dense_layer = Dense(units=self.num_states, activation='softmax')

    def call(self, init_hidden_state=None, targets=None):
        # === Embedding
        # targets: [None, target_steps]
        # embedding: [None, target_steps, embedding_dim]
        embedding = self.embedding_layer(targets)

        # encoder_hidden_state: [None, rnn_units]
        # outputs: [None, rnn_units]
        # decoder_hidden_state: [None, rnn_units]
        # outputs == decoder_hidden_state
        outputs, decoder_hidden_state = self.gru_layer(embedding, initial_state=init_hidden_state)

        # softmax: [None,num_states]
        softmax = self.dense_layer(decoder_hidden_state)

        return softmax, decoder_hidden_state


class Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size=10,
                 num_states=4,
                 embedding_dim=32,
                 rnn_units=16):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(vocab_size=vocab_size,
                               embedding_dim=embedding_dim,
                               rnn_units=rnn_units)

        self.decoder = Decoder(num_states=num_states,
                               embedding_dim=embedding_dim,
                               rnn_units=rnn_units)

    def call(self, inputs=None, targets=None):
        encoder_hidden_state = self.encoder(inputs=inputs)

        softmax, decoder_hidden_state = self.decoder(targets=targets,
                                                     init_hidden_state=encoder_hidden_state)

        outputs = []

        return softmax, decoder_hidden_state


def test_seq2seq_once():
    batch_size = 2
    steps = 7

    vocab_size = 10
    num_states = 4

    inputs = tf.random.uniform((batch_size, steps), minval=0, maxval=vocab_size, dtype=tf.int32)
    targets = tf.random.uniform((batch_size, steps), minval=0, maxval=num_states, dtype=tf.int32)

    print("encoder")
    encoder = Encoder(vocab_size=vocab_size)

    encoder_hidden_state = encoder(inputs=inputs)
    print(encoder_hidden_state.shape)

    print("decoder")
    decoder = Decoder(num_states=num_states)
    softmax, decoder_hidden_state = decoder(targets=targets, init_hidden_state=encoder_hidden_state)
    print('softmax', softmax.shape)
    print('decoder_hidden_state', decoder_hidden_state.shape)


    """
    print("rnn")
    rnn = Seq2Seq(vocab_size=vocab_size, num_states=num_states)

    softmax, decoder_hidden_state = rnn(inputs=inputs, targets=targets)
    print('softmax', softmax.shape)
    print('decoder_hidden_state', decoder_hidden_state.shape)
    """


if __name__ == "__main__":
    test_seq2seq_once()


