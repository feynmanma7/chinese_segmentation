from cangjie.utils.config import get_data_dir, get_model_dir
from cangjie.rnn.dataset import get_dataset
from cangjie.rnn.dictionary import load_dictionary
from cangjie.rnn.rnn import RNNSeg
from cangjie.utils.losses import mask_sparse_cross_entropy
import tensorflow as tf
import time
import os


def train_rnn_seg():
    vocab_size = 5168
    num_states = 4
    total_num_train = 69000
    total_num_val = 17300

    epochs = 100
    shuffle_buffer_size = 1024 * 2
    batch_size = 32
    rnn_steps = 30

    embedding_dim = 64
    rnn_units = 32
    pad_index = 0 # pad_index, to mask in loss

    train_path = os.path.join(get_data_dir(), "msr_rnn_train.utf8")
    val_path = os.path.join(get_data_dir(), "msr_rnn_val.utf8")
    word2id_dict_path = os.path.join(get_data_dir(), "msr_training_rnn_dict.pkl")

    num_train_batch = total_num_train // batch_size + 1
    num_val_batch = total_num_val // batch_size + 1

    word2id_dict = load_dictionary(dict_path=word2id_dict_path)
    print("#word2id_dict = %d" % len(word2id_dict))

    train_dataset = get_dataset(data_path=train_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size,
                                steps=rnn_steps,
                                char2id_dict=word2id_dict,
                                pad_index=pad_index)

    val_dataset = get_dataset(data_path=val_path,
                              epochs=epochs,
                              shuffle_buffer_size=shuffle_buffer_size,
                              batch_size=batch_size,
                              steps=rnn_steps,
                              char2id_dict=word2id_dict,
                              pad_index=pad_index)

    rnnseg = RNNSeg(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)

    optimizer = tf.keras.optimizers.Adam(0.001)

    checkpoint_dir = os.path.join(get_model_dir(), "rnn")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=rnnseg, optimizer=optimizer)

    # === Train
    #num_train_batch = 2 # only for debug
    #num_val_batch = 2

    min_val_loss = None
    opt_epoch = None
    patience = 5

    start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # === train
        print('\nTraining...')
        train_loss = 0
        for batch, (inputs, targets) in zip(range(num_train_batch), train_dataset):
            batch_start = time.time()

            cur_loss = train_step(rnnseg, optimizer, inputs, targets, mask=pad_index)
            train_loss += cur_loss

            if (batch+1) % 100 == 0:
                batch_end = time.time()
                batch_last = batch_end - batch_start

                print("Epoch: %d/%d, batch: %d/%d, train_loss: %.4f, cur_loss: %.4f, lasts: %.2fs"
                      % (epoch+1, epochs, batch+1, num_train_batch, train_loss/(batch+1), cur_loss, batch_last))

        train_loss /= num_train_batch
        print("Epoch: %d/%d, train_loss: %.4f"
              % (epoch+1, epochs, train_loss))

        # === validate
        print("\nValidating...")
        val_loss = 0
        for batch, (inputs, targets) in zip(range(num_val_batch), val_dataset):
            batch_start = time.time()

            cur_loss = train_step(rnnseg, optimizer, inputs, targets, mask=pad_index)
            val_loss += cur_loss

            if (batch+1) % 100 == 0:
                batch_end = time.time()
                batch_last = batch_end - batch_start

                print("Epoch: %d/%d, batch: %d/%d, val_loss: %.4f, cur_loss: %.4f, lasts: %.2fs"
                      % (epoch+1, epochs, batch, num_val_batch, val_loss/(batch+1), cur_loss, batch_last))

        val_loss /= num_val_batch

        epoch_end = time.time()
        epoch_last = epoch_end - epoch_start
        print("Epoch: %d/%d, train_loss: %.4f, val_loss: %.4f, lasts: %.2fs"
              % (epoch+1, epochs, train_loss, val_loss, epoch_last))

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
                  % (epoch+1, train_loss, val_loss))
            checkpoint.save(file_prefix=checkpoint_prefix)

    end = time.time()
    last = end - start
    print("Training done! min_val_loss=%.4f, opt_epoch=%d Lasts: %.2fs"
          % (min_val_loss, opt_epoch, last))


@tf.function
def train_step(rnnseg, optimizer, inputs, targets, mask=0):
    # inputs: [None, steps]
    # targets: [None, steps]

    with tf.GradientTape() as tape:
        # softmax: [None, steps, num_classes+1]
        softmax = rnnseg(inputs)

        # loss: shape=(), scalar
        loss = mask_sparse_cross_entropy(y_true=targets, y_pred=softmax, mask=mask)

    variables = rnnseg.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


if __name__ == "__main__":
    train_rnn_seg()