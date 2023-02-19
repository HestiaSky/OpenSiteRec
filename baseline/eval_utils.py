import numpy as np


def PrecisionRecall_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.mean(tp) / k
    recall_n = np.array([len(test[i]) if len(test[i]) > 0 else 1 for i in range(len(test))])
    recall = np.mean(tp / recall_n)
    return precision, recall


def NDCG_atK(test, r, k):
    pred = r[:, :k]
    test_mat = np.zeros((len(pred), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_mat[i, :length] = 1

    idcg = np.sum(test_mat * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    dcg = pred * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    ndcg = np.mean(ndcg)
    return ndcg


def get_label(test, pred):
    r = []
    for i in range(len(test)):
        groundTruth, predTopK = test[i], pred[i]
        hits = list(map(lambda x: x in groundTruth, predTopK))
        hits = np.array(hits).astype("float")
        r.append(hits)
    return np.array(r).astype('float')




