from cangjie.hmm.hmm import HMM
from cangjie.utils.config import get_data_dir, get_model_dir
import os


def seg_on_sentence(hmm, sentence):
    seg_words = hmm.decode(sentence=sentence)

    print("/".join(seg_words))


def seg_on_file(hmm, file_path):
    pass


if __name__ == "__main__":
    data_dir = get_data_dir()
    model_dir = get_model_dir()

    model_path = os.path.join(model_dir, "hmm", "hmm.pkl")
    test_path = os.path.join(data_dir, "msr_test.utf8")
    test_result_path = os.path.join(data_dir, "msr_test_hmm.utf8")

    hmm = HMM()
    hmm.load_model(model_path=model_path)

    seg_on_sentence(hmm, sentence='黑夜给了我黑色的眼睛，我却用它寻找光明。')

