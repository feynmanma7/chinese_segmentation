#encoding:utf-8
import os


def get_label(input_path, output_path):
    fw = open(output_path, 'w', encoding='utf-8')

    with open(input_path, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:

            words = []
            labels = []
            for word in line[:-1].split(' '):
                if word is None or len(word) == 0:
                    continue

                words.append(word)
                if len(word) == 1:
                    label = 'S'
                else:
                    label = ['M'] * len(word)
                    label[0] = 'B'
                    label[-1] = 'E'
                    label = ''.join(label)

                labels.append(label)

            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)

            fw.write(''.join(words) + '\t' + ''.join(labels) + '\n')

        print("Total", cnt)

    fw.close()


if __name__ == '__main__':
    data_dir = "/Users/flyingman/Developer/github/chinese_segmentation/data"

    input_path = os.path.join(data_dir, 'people.txt')
    output_path = os.path.join(data_dir, 'people_label.txt')

    get_label(input_path, output_path)