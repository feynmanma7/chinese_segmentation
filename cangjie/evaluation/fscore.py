import os


def to_index(buf):
    """
    buf:  word_0, word1, ..., word_{N-1}
    indexes(python style [start, end)):
        [start_0, end_0], [end_0, end_1], ..., [end_{N-2}, end_{N-1}=end]
    """

    indexes = []
    start = 0

    for word in buf:
        end = start + len(word)
        indexes.append((start, end))
        start = end

    return set(indexes)


def compute_num_intersect(buf_1, buf_2):
    if len(buf_1) == 0 or len(buf_2) == 0:
        return 0

    indexes_1 = to_index(buf_1)
    indexes_2 = to_index(buf_2)

    # Computing intersects between sets, using `&`.
    intersects = indexes_1 & indexes_2

    if intersects is not None:
        return len(intersects)

    return 0


def compute_fscore(y_true_path=None, y_pred_path=None):
    fr_true = open(y_true_path, 'r', encoding='utf-8')
    fr_pred = open(y_pred_path, 'r', encoding='utf-8')

    n_correct = 0
    n_true = 0
    n_pred = 0

    for true_line, pred_line in zip(fr_true, fr_pred):
        true_buf = []
        for word in true_line[:-1].split(' '):
            if len(word) == 0:
                continue
            true_buf.append(word)

        pred_buf = []
        for word in pred_line[:-1].split(' '):
            if len(word) == 0:
                continue
            pred_buf.append(word)

        cur_n_correct = compute_num_intersect(true_buf, pred_buf)

        n_correct += cur_n_correct
        n_true += len(true_buf)
        n_pred += len(pred_buf)

    if n_correct == 0:
        return False

    precision = n_correct / n_pred
    recall = n_correct / n_true
    f1_score = 2 * precision * recall / (precision + recall)

    print("precision\trecall\tf1_score")
    print("%.4f\t\t%.4f\t%.4f" %
          (precision, recall, f1_score))

    fr_true.close()
    fr_pred.close()

    return True


if __name__ == '__main__':
    data_dir = "/Users/flyingman/Developer/github/chinese_segmentation/data"

    method = "fmm"

    print(method)

    y_true_path = os.path.join(data_dir, "msr_test_gold.utf8")
    y_pred_path = os.path.join(data_dir, "msr_test_" + method + ".utf8")

    compute_fscore(y_true_path=y_true_path,
                   y_pred_path=y_pred_path)

