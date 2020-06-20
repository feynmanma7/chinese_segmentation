#encoding:utf-8
import os


def get_pre_char(char_seq, index):
    if index > 0:
        return '<>'
    return char_seq[index-1]


def get_next_char(char_seq, index):
    if index >= len(char_seq) - 1:
        return '</>'
    return char_seq[index+1]


def func_three_chars(char_seq, label_seq, index=None):
    pre_char = get_pre_char(char_seq, index)
    cur_char = char_seq[index]
    next_char = get_next_char(char_seq, index)

    return ','.join([pre_char, cur_char, next_char])


def feature_template():
    return [func_three_chars]


def generate_feature(char_seq, label_seq, index=None):
    feature_list = [func(char_seq, label_seq, index)
                    for func in feature_template()]

    return feature_list


def get_feature(input_path, feature_path, instance_path):
    """
    input_path: label:   char_seq \t label_seq
    feature_path:  feature \t cnt
    instance_path: label feature_1 :: feature_2 :: ... :: feature_N
    """
    fw_feat = open(feature_path, 'w', encoding='utf-8')
    fw_ins = open(instance_path, 'w', encoding='utf-8')

    feature_dict = {}

    line_cnt = 0
    ins_cnt = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            buf = line[:-1].split('\t')
            char_seq = buf[0]
            label_seq = buf[1]

            if len(char_seq) != len(label_seq):
                continue

            for i in range(len(char_seq)):
                label = label_seq[i]
                feature_list = generate_feature(char_seq, label_seq, index=i)

                # Count feature frequency.
                for feature in feature_list:
                    if feature not in feature_dict:
                        feature_dict[feature] = 1
                    else:
                        feature_dict[feature] += 1

                fw_ins.write(label + '\t' + '::'.join(feature_list) + '\n')
                ins_cnt += 1

            line_cnt = 0
            if line_cnt % 10000 == 0:
                print(line_cnt)

    fw_ins.close()

    print("Total lines is %d" % line_cnt)
    print("Total instances is %d" % ins_cnt)

    # Write feature_cnt dict to disk.
    for feature, cnt in feature_dict.items():
        fw_feat.write(str(feature) + '\t' + str(cnt) + '\n')
    fw_feat.close()

    print("Total feature is %d" % len(feature_dict))




if __name__ == "__main__":
    data_dir = "/Users/flyingman/Developer/github/chinese_segmentation/data"

    input_path = os.path.join(data_dir, 'people_label.txt')
    feature_path = os.path.join(data_dir, 'people_feature.txt')
    instance_path = os.path.join(data_dir, 'people_instance.txt')

    get_feature(input_path, feature_path, instance_path)


