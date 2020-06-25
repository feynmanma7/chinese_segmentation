from cangjie.hmm.hmm import HMM
from cangjie.utils.config import get_data_dir, get_model_dir
import os

if __name__ == '__main__':
    data_dir = get_data_dir()
    model_dir = get_model_dir()

    train_data_path = os.path.join(data_dir, "msr_training.utf8")
    model_path = os.path.join(model_dir, "hmm", "hmm.pkl")

    hmm = HMM()
    hmm.train(train_path=train_data_path, model_path=model_path)

