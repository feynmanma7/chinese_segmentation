import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
tf.random.set_seed(7)


def mask_sparse_cross_entropy(y_true=None, y_pred=None, mask=0):
    # y_true: [None, steps]
    # y_pred: [None, steps, num_classes+1], +1 for pad to mask

    # loss: [None, steps]
    loss = sparse_categorical_crossentropy(y_true, y_pred)

    # masks: [None, steps]
    masks = tf.cast(tf.not_equal(y_true, mask), tf.float32)

    # masked-loss: [None, steps]
    loss = tf.multiply(loss, masks)

    # reduce_mean: shape=()
    loss = tf.cast(tf.reduce_mean(loss), tf.float32)

    return loss


if __name__ == '__main__':
    batch_size = 2
    steps = 5
    num_classes = 4

    y_true = tf.random.uniform((batch_size, steps), minval=0, maxval=num_classes+1, dtype=tf.int32)
    y_pred = tf.random.uniform((batch_size, steps, num_classes+1))

    print('true')
    print(y_true)

    print('\npred')
    print(y_pred)

    loss = mask_sparse_cross_entropy(y_true=y_true, y_pred=y_pred, mask=0)
    print('loss')
    print(loss)
