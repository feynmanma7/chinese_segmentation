import jieba
from cangjie.utils.config import get_data_dir
import os

if __name__ == '__main__':
    test_path = os.path.join(get_data_dir(), "msr_test.utf8")
    seg_path = os.path.join(get_data_dir(), "msr_test_jieba.utf8")

    fw = open(seg_path, 'w', encoding='utf-8')
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            seg = jieba.cut(line[:-1])
            words = [word for word in seg]
            fw.write(" ".join(words) + '\n')

    fw.close()
    print("Write Done!", seg_path)
