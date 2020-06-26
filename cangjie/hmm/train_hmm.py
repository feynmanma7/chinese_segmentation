from cangjie.hmm.hmm import HMM
from cangjie.utils.config import get_data_dir, get_model_dir
import os

if __name__ == '__main__':
    data_dir = get_data_dir()
    model_dir = get_model_dir()

    model_path = os.path.join(model_dir, "hmm", "hmm.pkl")

    hmm = HMM()

    # train_data_path = os.path.join(data_dir, "msr_training.utf8")
    #hmm.train(train_path=train_data_path, model_path=model_path, is_incre_train=False)

    train_data_path = os.path.join(data_dir, "people.txt")
    hmm.train(train_path=train_data_path, model_path=model_path, is_incre_train=True)

