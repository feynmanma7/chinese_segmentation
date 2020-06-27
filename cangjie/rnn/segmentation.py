from cangjie.utils.config import get_model_dir, get_data_dir
import os




if __name__ == '__ain__':
    data_dir = get_data_dir()
    model_dir = get_model_dir()

    train_path = os.path.join(data_dir, "msr_training.utf8")
    model_path = os.path.join(model_dir, "seq2seq")



