from cangjie.utils.config import get_data_dir, get_model_dir, get_log_dir
from cangjie.rnn.dataset import get_dataset
from cangjie.rnn.dictionary import load_dictionary
from cangjie.rnn.rnn import RNNSeg
from cangjie.utils.losses import mask_sparse_cross_entropy
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import time
import os


def train_rnn_seg():
    vocab_size = 3954 # count > min_char_count = 5
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
    char2id_dict_path = os.path.join(get_data_dir(), "msr_training_char2id_dict.pkl")

    num_train_batch = total_num_train // batch_size + 1
    num_val_batch = total_num_val // batch_size + 1

    char2id_dict = load_dictionary(dict_path=char2id_dict_path)
    print("#char2id_dict = %d" % len(char2id_dict))

    train_dataset = get_dataset(data_path=train_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size,
                                steps=rnn_steps,
                                char2id_dict=char2id_dict,
                                pad_index=pad_index)

    val_dataset = get_dataset(data_path=val_path,
                              epochs=epochs,
                              shuffle_buffer_size=shuffle_buffer_size,
                              batch_size=batch_size,
                              steps=rnn_steps,
                              char2id_dict=char2id_dict,
                              pad_index=pad_index)

    rnnseg = RNNSeg(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
    optimizer = tf.keras.optimizers.Adam(0.001)

    rnnseg.compile(optimizer=optimizer,
                   loss=mask_sparse_cross_entropy,
                   metrics=['acc'])

    callbacks = []

    early_stopping_cb = EarlyStopping(monitor='val_loss',
                                    patience=5, restore_best_weights=True)
    callbacks.append(early_stopping_cb)

    tensorboard_cb = TensorBoard(log_dir=os.path.join(get_log_dir(), "rnn_model"))
    callbacks.append(tensorboard_cb)

    checkpoint_path = os.path.join(get_model_dir(), "rnn_model", "ckpt")
    checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True,
                                    save_best_only=True)
    callbacks.append(checkpoint_cb)

    history = rnnseg.fit(train_dataset,
               batch_size=batch_size,
               epochs=epochs,
               steps_per_epoch=num_train_batch,
               validation_data=val_dataset,
               validation_steps=num_val_batch,
               callbacks=callbacks)

    print(rnnseg.summary())

    return True


if __name__ == "__main__":
    start = time.time()
    train_rnn_seg()
    end = time.time()
    last = end - start
    print("\nTrain done! Lasts: %.2fs" % last)