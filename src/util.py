import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import stats




def parse_response(str_id):
    if str_id.split('_')[1] == 'resp':
        return 1
    return 0


def preprocess_dataset(dataset):
    gen_names = dataset["SYMBOL"].as_matrix()
    responses = np.zeros(dataset.shape[1] - 1, dtype=np.int)
    target_ids = dataset.columns[1:].values
    feature_matrix = np.zeros((dataset.shape[1] - 1, dataset.shape[0]), dtype=np.float)
    for idx, target_id in enumerate(target_ids):
        feature_matrix[idx, :] = dataset[target_id].values
        responses[idx] = parse_response(target_id)
    return feature_matrix, responses, target_ids, gen_names


# def get_relation_matrix(order_seq):
#     matrix = np.zeros(shape=(order_seq.shape[0], order_seq.shape[0]), dtype=np.int32)
#     for row in range(matrix.shape[0]):
#         for col in range(matrix.shape[1]):
#             matrix[row, col] = order_seq[row] - order[col]
#     return matrix


# def get_relation_cumul_matrix(order_matrix):
#     matrix = np.zeros(shape=(order_matrix.shape[1], order_matrix.shape[1]), dtype=np.float32)
#     for row in range(matrix.shape[1]):
#         for col in range(matrix.shape[1]):
#             matrix[row, col] = (order_matrix[:, row] - order_matrix[:, col]).mean()
#         if row % 1000 == 0:
#             print(row)
#     return matrix


def get_metrics(conf):
    """return tuple (precision 0 class, recall 0 class, precision 1 class, recall 1 class)"""
    prec_0 = conf[0, 0] / (conf[1, 0] + conf[0, 0])
    if (conf[1, 0] + conf[0, 0]) == 0:
        prec_0 = 0
    recall_0 = conf[0, 0] / (conf[0, 0] + conf[0, 1])
    prec_1 = conf[1, 1] / (conf[1, 1] + conf[0, 1])
    recall_1 = conf[1, 1] / (conf[1, 1] + conf[1, 0])
    if (conf[1, 1] + conf[0, 1]) == 0:
        prec_1 = 0
    accuracy = (conf[0, 0] + conf[1, 1]) / (conf.sum())
    return prec_0, recall_0, prec_1, recall_1


def gen_samples(X, y, iters=10, cv=10, bootstrap_max=False):
    samples = []
    for it in range(iters):
        subsample = get_balanced_subsample(y, bootstrap_max=bootstrap_max)
        skfold = StratifiedKFold(n_splits=cv, shuffle=True)
        skfold.get_n_splits(X[subsample], y[subsample])  # для совместимости с предыдущей версией
        samples.append([(subsample[train], subsample[test])
                        for train, test in skfold.split(X[subsample], y[subsample])])
    return samples


def get_balanced_subsample(y, bootstrap_max=False):
    assert np.unique(y).shape[0] == 2
    nonresp = np.arange(y.shape[0])[y == 0]
    resp = np.arange(y.shape[0])[y == 1]
    np.random.shuffle(resp)
    np.random.shuffle(nonresp)
    if not bootstrap_max:
        class_size = min(nonresp.shape[0], resp.shape[0])
        return np.hstack((nonresp[:class_size], resp[:class_size]))
    class_size = max(nonresp.shape[0], resp.shape[0])
    nr_addition = np.random.choice(nonresp, size=class_size - nonresp.shape[0])
    r_addition = np.random.choice(resp, size=class_size - resp.shape[0])
    return np.hstack((nonresp, nr_addition, resp, r_addition))


def do_simmetry_class_cv(X, y, samples, clf, feature_maker):
    results_a, results_b, cv_results, accs = [], [], [], []
    all_descriptions = []
    for ind, sample in enumerate(samples):
        print("iteration: {}".format(ind))
        stat, common_stat, cv_result, descriptions = do_cross_val(X, y, sample,
                                                                  clf, feature_maker)
        results_a.append(stat.mean(axis=0))
        results_b.append(np.array(common_stat))
        cv_results.append(cv_result)
        all_descriptions.append(descriptions)
    return np.vstack(results_a), np.vstack(results_b), cv_results, all_descriptions


def do_cross_val(X, y, sample, clf, feature_maker):
    cv_results, descriptions = [], []
    for train, test in sample:
        features, description = feature_maker(X, y, train, test)
        descriptions.append(description)
        clf.fit(features[train, :], y[train])
        prediction = clf.predict(features[test, :])
        proba = clf.predict_proba(features[test, :])
        cv_results.append((y[test], prediction, proba[:, 1]))
    conf_m = confusion_matrix(cv_results[0][0], cv_results[1][0]) * 0
    stat, fpr_tpr, tpr, accs = [], [], [], []
    for actual, predicted, proba in cv_results:
        conf = confusion_matrix(actual, predicted)
        stat.append(np.array(get_metrics(conf)))
        conf_m = conf_m + conf
    return np.vstack(stat), get_metrics(conf_m), cv_results, descriptions


def process_results(a, b, cv_results):
    a_aucs, a_accs, b_aucs, b_accs = [], [], [], []
    for result in cv_results:
        actuals, preds, probs = [], [], []
        for actual, predicted, proba in result:
            a_accs.append((actual == predicted).mean())
            actuals.append(actual)
            preds.append(predicted)
            probs.append(proba)
            fpr, tpr, _ = roc_curve(actual, proba)
            a_aucs.append(auc(fpr, tpr))
        b_accs.append((np.hstack(actuals) == np.hstack(preds)).mean())
        fpr, tpr, thresh = roc_curve(np.hstack(actuals), np.hstack(probs))
        b_aucs.append(auc(fpr, tpr))
    columns = ["precision 0", "recall 0", "precision 1", "recall 1", "accuracy", "auc"]
    rows = ["схема а", "схема б"]
    data = np.zeros((len(rows), len(columns)))
    data[0, :4] = a.mean(axis=0)
    data[0, 4:] = [np.mean(a_accs), np.mean(a_aucs)]
    data[1, :4] = b.mean(axis=0)
    data[1, 4:] = [np.mean(b_accs), np.mean(b_aucs)]
    table = pd.DataFrame(data, columns=columns, index=rows)
    return table


def read_GEO_dataset(pathto_xpr, pathto_resp, pathto_nonresp, prefix=""):
    geo_dataset = pd.read_csv(os.path.join(prefix, pathto_xpr), sep=" ")
    with open(os.path.join(prefix, pathto_resp), "r") as respf, \
            open(os.path.join(prefix, pathto_nonresp), "r") as nonrespf:
        resp = set(respf.read().split())
        nonresp = set(nonrespf.read().split())
    responses = np.array(list(map(lambda x: 1 if x in resp else int(x in nonresp) - 1,
                                  geo_dataset.columns[1:])))
    feature_matrix = geo_dataset[geo_dataset.columns[1:]].get_values().T
    target_ids = geo_dataset.columns[1:].values
    gen_names = geo_dataset["Group.1"].as_matrix()
    return feature_matrix, responses, target_ids, gen_names


def test_auc_greater(cv_0, cv_1):
    aucs_0 = []
    aucs_1 = []
    for result in cv_0:
        for actual, predicted, proba in result:
            fpr, tpr, _ = roc_curve(actual, proba)
            aucs_0.append(auc(fpr, tpr))
    for result in cv_1:
        for actual, predicted, proba in result:
            fpr, tpr, _ = roc_curve(actual, proba)
            aucs_1.append(auc(fpr, tpr))
    aucs_0, aucs_1 = np.array(aucs_0), np.array(aucs_1)
    return stats.mannwhitneyu(aucs_0, aucs_1, alternative='less').pvalue


def do_simmetry_class_cv_by_descr(X, y, samples, descriptions, clf, feature_maker):
    results_a, results_b, cv_results, accs = [], [], [], []
    for ind, sample in enumerate(samples):
        print("iteration: {}".format(ind))
        stat, common_stat, cv_result, _ = \
        do_cross_val_by_descr(X, y, sample, descriptions[ind],
                              clf, feature_maker)
        results_a.append(stat.mean(axis=0))
        results_b.append(np.array(common_stat))
        cv_results.append(cv_result)
    return np.vstack(results_a), np.vstack(results_b), cv_results, descriptions


def do_cross_val_by_descr(X, y, sample, descriptions, clf, feature_maker):
    cv_results = []
    ind = 0
    for train, test in sample:
        features, _ = feature_maker(X, y, train, test, descriptions[ind])
        clf.fit(features[train, :], y[train])
        prediction = clf.predict(features[test, :])
        proba = clf.predict_proba(features[test, :])
        cv_results.append((y[test], prediction, proba[:, 1]))
        ind += 1
    conf_m = confusion_matrix(cv_results[0][0], cv_results[1][0]) * 0
    stat, fpr_tpr, tpr, accs = [], [], [], []
    for actual, predicted, proba in cv_results:
        conf = confusion_matrix(actual, predicted)
        stat.append(np.array(get_metrics(conf)))
        conf_m = conf_m + conf
    return np.vstack(stat), get_metrics(conf_m), cv_results, descriptions