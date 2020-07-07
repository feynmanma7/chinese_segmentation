from cangjie.utils.config import get_data_dir
import numpy as np
import os


def split_train_val(input_path=None,
                    train_path=None,
                    val_path=None,
                    train_ratio=None):
    fw_train = open(train_path, 'w', encoding='utf-8')
    fw_val = open(val_path, 'w', encoding='utf-8')

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = np.random.random()
            if r < train_ratio:
                fw_train.write(line)
            else:
                fw_val.write(line)

    fw_train.close()
    fw_val.close()


if __name__ == '__main__':
    input_path = os.path.join(get_data_dir(), "msr_training_label.utf8")
    train_path = os.path.join(get_data_dir(), "msr_rnn_train.utf8")
    val_path = os.path.join(get_data_dir(), "msr_rnn_val.utf8")

    train_ratio = 0.8

    split_train_val(input_path=input_path,
                    train_path=train_path,
                    val_path=val_path,
                    train_ratio=train_ratio)

    print("Write done!", train_path)
