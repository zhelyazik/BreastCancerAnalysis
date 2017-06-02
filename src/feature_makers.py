import numpy as np
from sklearn.linear_model import LogisticRegression




def xpressions_to_order_matrix(xpressions):
    orders_num = np.arange(xpressions.shape[1], dtype=np.int32)
    order_matrix = np.zeros_like(xpressions, dtype=np.int32)
    for row in range(xpressions.shape[0]):
        ordered_ids = np.array([ind
                                for ind, xpr in sorted([(ind, xpr)
                                                        for ind, xpr in enumerate(xpressions[row, :])],
                                                       key=lambda p: p[1])])
        order = np.zeros_like(ordered_ids, dtype=np.int32)
        order[ordered_ids] = np.arange(xpressions.shape[1])
        order_matrix[row, :] = order
    return order_matrix


def xpression_to_order_feature(xpressions, pairs, gens):
    order_features = np.zeros(shape=(xpressions.shape[0], len(pairs)))
    features_name = []
    for idx, pair in enumerate(pairs):
        order_features[:, idx] = xpressions[:, pair[0]] > xpressions[:, pair[1]] + 0
        features_name.append("compare_expr({}, {})".format(gens[pair[0]], gens[pair[1]]))
    return order_features, features_name


def get_n_biggest_sims_ids(sims, subset, n=140):
    subsims = sims[subset]
    bound = np.sort(subsims)[-n]
    return subset[subsims >= bound][:n]


def get_gen_hists(xpression, cl_masks, bins=10):
    xpr_range = (xpression.min(), xpression.max())
    hists = []
    for mask in cl_masks:
        hist, _ = np.histogram(xpression[mask], bins=bins, range=xpr_range)
        hists.append(hist)
    return hists


def get_pdf_sim(p, q):
    return -q.dot(np.log2(p)) - p.dot(np.log2(q))


def heuristic_1_2_features(X, y, train, test, topn=250):
    #description = "экспрессии генов отобранных по пересечению 1-й и второй эвристик\n id генов:\n"
    gen_res = []
    for g in range(X.shape[1]):
        lg = LogisticRegression(penalty='l1', C=1)
        lg.fit(X[train, g:g + 1], y[train])
        gen_res.append((lg.predict(X[train, g:g + 1]) == y[train]).mean())
    gen_res = np.array(gen_res)
    good_h1 = np.arange(X.shape[1])[gen_res > y[train].mean()]

    gen_corr = []
    for g in range(X.shape[1]):
        gen_corr.append(abs(np.corrcoef(X[train, g], y[train])[0, 1]))
    gen_corr = np.array(gen_corr)
    gen_corr_indexed = sorted(list(enumerate(gen_corr)), key=lambda p: -p[1])
    good = []
    for i, g in gen_corr_indexed:
        if i in good_h1:
            good.append(i)
        if len(good) == topn:
            break
            
    good = np.array(good)   
    description = "\t".join([str(gid) for gid in good])
    return X[:, good], description


def get_pair_value(order_matrix, pair, responses, pr=None):
    if not pr:
        pr = (responses == 1).mean()
    p_ab = np.mean(order_matrix[:, pair[0]] > order_matrix[:, pair[1]])
    p_ab_r = np.mean(order_matrix[responses == 1, pair[0]] > order_matrix[responses == 1, pair[1]])
    pr_ab = (p_ab_r * pr) / p_ab
    p_ba = np.mean(order_matrix[:, pair[0]] < order_matrix[:, pair[1]])
    p_ba_r = np.mean(order_matrix[responses == 1, pair[0]] < order_matrix[responses == 1, pair[1]])
    pr_ba = (p_ba_r * pr) / p_ba
    return pr_ab, pr_ba


def heuristic_4_features(X, y, train, test, topn=70):
    print("heuristic 4 started")
    order_matrix = xpressions_to_order_matrix(X[train, :])
    value_matrix = np.zeros(shape=(order_matrix.shape[1],
                                   order_matrix.shape[1]),
                            dtype=np.float32)
    for i in range(order_matrix.shape[1]):
        for j in range(i + 1, order_matrix.shape[1]):
            pr_ab, pr_ba = get_pair_value(order_matrix, (i, j), y[train])
            value_matrix[i, j] = pr_ab
            value_matrix[j, i] = pr_ba
    if i % 5000 == 0:
        print("\tcomplete: {} rows".format(str(i)))
    new_val = value_matrix + 0.
    for i in range(order_matrix.shape[1]):
        for j in range(order_matrix.shape[1]):
            if np.isnan(value_matrix[i, j]):
                new_val[i, j] = 0.
                new_val[j, i] = 0.
    dif_matrix = new_val - new_val.T
    p = np.percentile(dif_matrix, 99.99)
    pairs_val = []
    for i in range(order_matrix.shape[1]):
        for j in range(order_matrix.shape[1]):
            if dif_matrix[i, j] >= p:
                pairs_val.append(((i, j), dif_matrix[i, j]))
    pairs_val = sorted(pairs_val, key=lambda p_v: -p_v[1])
    good_pairs = [p_v[0] for p_v in pairs_val[:topn]]
    order_features, pair_names = xpression_to_order_feature(X, good_pairs, np.arange(X.shape[1]))
    description = "\t".join(pair_names)
    return order_features, description


def heuristic_4_rnd(X, y, train, test, topn=70):
    # print("heuristic 4 started")
    good_pairs = []
    choosen = np.random.choice(np.arange(X.shape[1]), size=2 * topn, replace=False)
    for i in range(0, 2 * topn, 2):
        good_pairs.append((choosen[i], choosen[i + 1]))
    order_features, pair_names = xpression_to_order_feature(X, good_pairs, np.arange(X.shape[1]))
    description = "\t".join(pair_names)
    return order_features, description


def identity_heuristic(X, y, train, test):
    return X, 'identity'


def rnd_gen_heuristic(X, y, train, test, topn=250):
    gids = np.arange(X.shape[0])
    np.random.shuffle(gids)
    return X[:, gids[:topn]], '\t'.join([str(gid) for gid in gids[:topn]])


def heuristic_3(X, y, train, test, topn=250):
    resp_mask = y[train] == 1
    nonresp_mask = y[train] == 0
    gen_pdfs = []
    nb_bins = 15
    for g in range(X[train].shape[1]):
        hists = get_gen_hists(X[train][:, g], [resp_mask, nonresp_mask], bins=nb_bins)
        p = (hists[0] + 1) / (hists[0].sum() + hists[0].shape[0])
        q = (hists[1] + 1) / (hists[1].sum() + hists[1].shape[0])
        gen_pdfs.append((p, q))
    gen_pdf_sims = []
    for gen_pdf in gen_pdfs:
        gen_pdf_sims.append(get_pdf_sim(gen_pdf[0], gen_pdf[1]))
    gen_pdf_sims = np.array(gen_pdf_sims)
    good_h3 = get_n_biggest_sims_ids(gen_pdf_sims, np.arange(X.shape[0]), 250)
    # print('complete')
    return X[:, good_h3], '\t'.join([str(g) for g in good_h3])


def heuristic_4_by_description(X, y, train, test, description):
    good_pairs = [tuple((int(g) for g in p.replace("compare_expr(", "").replace(")", "").split(", ")))
                  for p in description.split("\t")]
    order_features, _ = xpression_to_order_feature(X, good_pairs, np.arange(X.shape[1]))
    return order_features, description


def heuristic_3_by_description(X, y, train, test, description):
    good_h3 = [int(g) for g in description.strip().split('\t')]
    return X[:, good_h3], description

