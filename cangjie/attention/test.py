import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    a = tf.random.uniform((2, 2, 3))
    print(a.shape)
    print('a', a)
    b = tf.nn.softmax(a)
    print('b', b)
    print('1', tf.nn.softmax(a, axis=1))
    print('2', tf.nn.softmax(a, axis=2))

