import numpy as np
import math
import scipy
import multiprocessing as mp
from path_analysis import get_pathcount
from functools import partial


def get_pathcount_forest_single(rfc, n_features):
    countmatrix = np.zeros((n_features, n_features), dtype=np.int64)
    for num in range(len(rfc.estimators_)):
        countmatrix += get_pathcount_forest(rfc, num)
    return countmatrix

def get_pathcount_forest_async(rfc, n_features):
    mp_countmatrices = []
    pool = mp.Pool(mp.cpu_count())
    for i in range(len(rfc.estimators_)):
        mp_countmatrices.append(pool.apply_async(get_pathcount_forest, args = (rfc, i)))
    pool.close()
    pool.join()
    countmatrix = sum(res.get() for res in mp_countmatrices)
    return countmatrix


def get_pathcount_forest_pool(rfc, n_features):
    pool = mp.Pool(mp.cpu_count())
    func = partial(get_pathcount_forest, rfc)
    mp_countmatrices = pool.map(func, range(len(rfc.estimators_)))
    pool.close()
    pool.join()
    countmatrix = sum(x for x in mp_countmatrices)
    return countmatrix

def get_pathcount_forest(rfc, num):
    children_left = np.array(rfc.estimators_[num].tree_.children_left, dtype=np.int64)
    children_right = np.array(rfc.estimators_[num].tree_.children_right, dtype=np.int64)
    feature = np.array(rfc.estimators_[num].tree_.feature, dtype=np.int64)
    countmatrix = get_pathcount(children_left,children_right,feature, 1000)
    return countmatrix


def correct_countmatrix_bi_formula(countmatrix_uni, root_count, rfc):
    countmatrix_bi = countmatrix_uni + np.transpose(countmatrix_uni)

    max_depth = []
    for tree_in_forest in rfc.estimators_:
        max_depth.append(tree_in_forest.tree_.max_depth)

        mean_max_depth = np.round(np.mean(max_depth))
    mean_num_paths = np.round(np.mean(root_count))
    rand_chance = (1 / rfc.n_features_ ^ 2) * (
            math.factorial(mean_max_depth) / (math.factorial(mean_max_depth - 2))) * mean_num_paths
    countmatrix_bi_corr = scipy.sparse.csr_matrix.todense(countmatrix_bi) - rand_chance
    return countmatrix_bi_corr


def correct_countmatrix_bi_formula_precise(countmatrix_uni, num_paths, rfc):
    countmatrix_bi = countmatrix_uni + np.transpose(countmatrix_uni)

    max_depth = []
    for tree_in_forest in rfc.estimators_:
        max_depth.append(tree_in_forest.tree_.max_depth)
    chancesamepath = 0
    chancenode = (1 / rfc.n_features_ ^ 2)
    for x in range(len(max_depth)):
        chancesamepath += math.factorial(max_depth[x]) / (math.factorial(max_depth[x] - 2)) * num_paths[x]

    return scipy.sparse.csr_matrix.todense(countmatrix_bi) - chancenode * chancesamepath


def main():
    print('test')

if __name__ == '__main__':
    main()
