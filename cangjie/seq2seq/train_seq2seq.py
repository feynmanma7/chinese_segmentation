from cangjie.rnn.dictionary import load_dictionary # same with rnn

from cangjie.utils.config import get_data_dir, get_model_dir
from cangjie.seq2seq.dataset import get_dataset # different from rnn
from cangjie.seq2seq.seq2seq import Encoder, Decoder
from cangjie.utils.losses import mask_sparse_cross_entropy
import tensorflow as tf
import time
import os


def train_seq2seq():
    vocab_size = 3954 # count > min_char_count = 5
    num_states = 4
    total_num_train = 69000 # num_lines of msr_rnn_train.utf8
    total_num_val = 17300  # num_lines of msr_rnn_val.utf8
    batch_size = 32

    epochs = 100
    shuffle_buffer_size = 1024 * 2
    rnn_steps = 30

    embedding_dim = 64
    rnn_units = 32

    min_val_loss = None
    opt_epoch = None
    patience = 5

    train_path = os.path.join(get_data_dir(), "msr_rnn_train.utf8")
    val_path = os.path.join(get_data_dir(), "msr_rnn_val.utf8")
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")

    num_train_batch = total_num_train // batch_size + 1
    num_val_batch = total_num_val // batch_size + 1

    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict = %d" % len(char2id_dict))

    # === Dataset
    train_dataset = get_dataset(data_path=train_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size,
                                steps=rnn_steps,
                                char2id_dict=char2id_dict,
                                pad_index=0)

    val_dataset = get_dataset(data_path=val_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size,
                                steps=rnn_steps,
                                char2id_dict=char2id_dict,
                                pad_index=0)

    # === Model
    encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    decoder = Decoder(num_states=num_states, embedding_dim=embedding_dim, rnn_units=rnn_units)

    # === Optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)

    # === Checkpoint
    checkpoint_dir = os.path.join(get_model_dir(), "seq2seq")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)

    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()

        # === train
        print('\nTraining...')
        train_loss = 0

        batch_start = time.time()
        for batch, (inputs, targets) in zip(range(num_train_batch), train_dataset):
            cur_loss = train_step(encoder, decoder, optimizer, inputs, targets, mask=0)
            train_loss += cur_loss

            if (batch+1) % 100 == 0:
                print("Epoch: %d/%d, batch: %d/%d, train_loss: %.4f, cur_loss: %.4f,"
                      % (epoch+1, epochs, batch+1, num_train_batch, train_loss/(batch+1), cur_loss),
                      end=" ")

                batch_end = time.time()
                batch_last = batch_end - batch_start
                print("lasts: %.2fs" % batch_last)

        train_loss /= num_train_batch
        print("Epoch: %d/%d, train_loss: %.4f"
              % (epoch + 1, epochs, train_loss))

        # === validate
        print("\nValidating...")
        val_loss = 0

        batch_start = time.time()
        for batch, (inputs, targets) in zip(range(num_val_batch), val_dataset):
            cur_loss = train_step(encoder, decoder, optimizer, inputs, targets, mask=0)
            val_loss += cur_loss

            if (batch+1) % 100 == 0:
                print("Epoch: %d/%d, batch: %d/%d, val_loss: %.4f, cur_loss: %.4f, "
                      % (epoch+1, epochs, batch+1, num_val_batch, val_loss/(batch+1), cur_loss),
                      end=" ")

                batch_end = time.time()
                batch_last = batch_end - batch_start
                print("lasts: %.2fs" % batch_last)

        val_loss /= num_val_batch
        print("Epoch: %d/%d, train_loss: %.4f, val_loss: %.4f, "
              % (epoch+1, epochs, train_loss, val_loss),
              end=" ")

        epoch_end = time.time()
        epoch_last = epoch_end - epoch_start
        print("lasts: %.2fs" % epoch_last)

        if opt_epoch is not None:
            if epoch - opt_epoch > patience:
                print("Stop training, epoch: %d, opt_epoch: %d")
                break

        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            opt_epoch = epoch

            # === Save best model only.
            print("\nSaving...")
            print("Epoch: %d, train_loss: %.4f, val_loss: %.4f"
                  % (epoch + 1, train_loss, val_loss))
            checkpoint.save(file_prefix=checkpoint_prefix)


    print("Training done! min_val_loss=%.4f, opt_epoch=%d"
          % (min_val_loss, opt_epoch), end=" ")
    end = time.time()
    last = end - start
    print("Lasts: %.2fs" % last)


"""
If the decorator of tf.function is open, a bug below will be caused.
`ValueError: tf.function-decorated function tried to create variables on non-first call.`
"""
#@tf.function
def train_step(encoder, decoder, optimizer, inputs, targets, mask=0):
    # inputs: [None, steps]
    # targets: [None, steps]
    loss = 0.

    with tf.GradientTape() as tape:
        # === Teacher-force learning
        encoder_hidden_state = encoder(inputs=inputs)

        # targets[:, 0], [None, ]
        # target_start: [None, 1]
        target_start = tf.expand_dims(targets[:, 0], axis=1)

        # decoder_output: [None, 1, num_states+1]
        # decoder_state: [None, 1, rnn_units]
        decoder_output, decoder_state = \
            decoder(targets=target_start, pre_hidden_state=encoder_hidden_state)

        for t in range(1, int(targets.shape[1])):
            # targets[:, t], [None, ]
            # decoder_target: [None, 1]
            decoder_target = tf.expand_dims(targets[:, t], axis=1)

            loss += mask_sparse_cross_entropy(
                y_true=decoder_target, y_pred=decoder_output, mask=mask)

            # decoder_output: [None, 1, num_states+1]
            # decoder_state: [None, rnn_units]
            decoder_output, decoder_state = decoder(targets=decoder_target,
                                                    pre_hidden_state=decoder_state)

        batch_loss = loss / int(targets.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def test_train_step():
    vocab_size = 3954  # count > min_char_count = 5
    num_states = 4
    batch_size = 2

    rnn_steps = 10
    embedding_dim = 32
    rnn_units = 16

    encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    decoder = Decoder(num_states=num_states, embedding_dim=embedding_dim, rnn_units=rnn_units)

    optimizer = tf.keras.optimizers.Adam(0.001)

    inputs = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=vocab_size + 2, dtype=tf.int32)
    targets = tf.random.uniform((batch_size, rnn_steps), minval=0, maxval=num_states + 1, dtype=tf.int32)

    loss = train_step(encoder, decoder, optimizer, inputs, targets, mask=0)
    print(loss)


if __name__ == "__main__":
    test_train_step()
    train_seq2seq()