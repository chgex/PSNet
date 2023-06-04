

import numpy as np


def compute_e1(y_true, y_pred):
    """
    compute E1, e XOR operation to evaluate the inconsistent pixels between pred MASK and gt MASK
    range is [0, 1], the smaller, the better
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.array(y_true, dtype="int")
    y_pred = np.array(y_pred, dtype="int")
    assert y_true.shape == y_pred.shape and len(y_true.shape) == 2

    e1 = np.logical_xor(y_true, y_pred)
    e1 = np.mean(e1)

    return e1


def compute_F1_socre(y_true, y_pred):
    """
    compute e2, precision and recall
    e2: the samller, the better, precision and recall: the higher, the better.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.array(y_true, dtype="int")
    y_pred = np.array(y_pred, dtype="int")

    assert y_true.shape == y_pred.shape and len(y_true.shape) == 2

    tp, fp, tn, fn = 0, 0, 0, 0

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    e2 = (fp + fn) / (len(y_true) * 2)

    if tp == 0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2*tp / (2*tp + fp + fn)

    return e2, f1_score, precision, recall


if __name__ == "__main__":

    a = b = np.zeros((4, 4), dtype='int')
    # b = np.ones((4, 4), dtype='int')

    a[1, :] = 1
    b[1, :] = 1


    e1 = compute_e1(a, b)
    e2, f1, p, r = compute_F1_socre(a, b)

    print(e1, e2)
    print(f1, p, r)

    # a = a.reshape(-1)
    # print(a.shape[0])
    # # print(a)
    # print(len(a))
    # print(a[0])


