import numpy as np
import math
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import scipy
from multiprocessing import Pool
from joblib import Parallel, delayed







def get_pathcount(rfc, n_features):
    countmatrix = np.zeros([n_features, n_features])
    all_rootcount = np.zeros(len(rfc.estimators_))
    x = 0
    for estimator in rfc.estimators_:
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature

        seen = []
        rootcount = 0
        node_list = []
        for node_number in range(len(children_left) - 1, -1, -1):  # start in most likely leaf
            if children_right[node_number] == -1 and children_left[node_number] == -1 and children_right[node_number] not in seen:  # if leaf and not earlier seen
                while node_number != 0:  # while we have not reached the root node
                    parent_left = np.where(children_left == node_number)[0]  # check the parent left
                    parent_right = np.where(children_right == node_number)[0]  # check the parent right
                    if parent_left.size > 0:
                        parent_left = np.int(parent_left)
                        if feature[node_number] > -1:
                            node_list.append(node_number)
                            for i in node_list:
                                countmatrix[feature[parent_left], feature[i]] += 1  # count
                        node_number = parent_left  # iterate until reach root node
                    elif parent_right.size > 0:
                        parent_right = np.int(parent_right)
                        if feature[node_number] > -1:
                            node_list.append(node_number)
                            for i in node_list:
                                countmatrix[feature[parent_right], feature[i]] += 1
                        node_number = parent_right
                rootcount += 1
                node_list = []
            seen.append(node_number)

        all_rootcount[x] = rootcount
        x = x + 1

    return countmatrix, all_rootcount


# TODO: make pathcount with parents only, and check! Rewrite top-down for speed.

def get_pathcount_parentonly(rfc, n_features, verbose = 1):
    countmatrix = np.zeros([n_features, n_features])
    forest_size =len(rfc.estimators_)
    x = 0
    for estimator in rfc.estimators_:
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature

        seen = []
        for node_number in range(len(children_left) - 1, -1, -1):  # start in most likely leaf
            #print(node_number)
            if children_right[node_number] == -1 and children_left[node_number] == -1 and children_right[node_number] not in seen:  # if leaf and not earlier seen
                while node_number != 0:  # while we have not reached the root node
                    parent_left = np.where(children_left == node_number)[0]  # check the parent left
                    parent_right = np.where(children_right == node_number)[0]  # check the parent right
                    if parent_left.size > 0:
                        parent_left = np.int(parent_left)
                        if feature[node_number] > -1:
                            countmatrix[feature[parent_left], feature[node_number]] += 1  # count
                        node_number = parent_left  # iterate until reach root node
                    elif parent_right.size > 0:
                        parent_right = np.int(parent_right)
                        if feature[node_number] > -1:
                            countmatrix[feature[parent_right], feature[node_number]] += 1
                        node_number = parent_right
            seen.append(node_number)
        if verbose:
            print("done " + str(x) + " of " + str(forest_size))
        x = x + 1

    return countmatrix



def get_pathcount_boosting(rfc, n_features, verbose = 1):
    #countmatrix = np.zeros([n_features, n_features])
    countmatrix = scipy.sparse.csr_matrix([n_features, n_features])
    forest_size =len(rfc.estimators_)
    x = 0
    for k in range(forest_size):
        children_left = rfc.estimators_[k][0].tree_.children_left
        children_right = rfc.estimators_[k][0].tree_.children_right
        feature = rfc.estimators_[k][0].tree_.feature

        seen = []
        for node_number in range(len(children_left) - 1, -1, -1):  # start in most likely leaf
            #print(node_number)
            if children_right[node_number] == -1 and children_left[node_number] == -1 and children_right[node_number] not in seen:  # if leaf and not earlier seen
                while node_number != 0:  # while we have not reached the root node
                    parent_left = np.where(children_left == node_number)[0]  # check the parent left
                    parent_right = np.where(children_right == node_number)[0]  # check the parent right
                    if parent_left.size > 0:
                        parent_left = np.int(parent_left)
                        if feature[node_number] > -1:
                            countmatrix[feature[parent_left], feature[node_number]] += 1  # count
                        node_number = parent_left  # iterate until reach root node
                    elif parent_right.size > 0:
                        parent_right = np.int(parent_right)
                        if feature[node_number] > -1:
                            countmatrix[feature[parent_right], feature[node_number]] += 1
                        node_number = parent_right
            seen.append(node_number)
        if verbose:
            print("done " + str(x) + " of " + str(forest_size))
        x = x + 1

    return countmatrix

# def pathcountv2_parentonly(rfc,n_features, njobs = -1):
#     # """optimized version of pathcount"""
#     # countmatrix = np.zeros([n_features, n_features])
#     # forest_size =len(rfc.estimators_)
#     # x = 0
#     # pool = Pool(25)
#     # rfcit = [estimator1, estimator2...]
#     # countmatrix_single = pool.map(pathcount_single_parentonly(estimator,n_features))
#     # x +=1
#
#     return countmatrix


def pathcount_single_parentonly(estimator, n_features, x):
    countmatrix = np.zeros([n_features, n_features])
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature

    seen = []
    for node_number in range(len(children_left) - 1, -1, -1):  # start in most likely leaf
        #print(node_number)
        if children_right[node_number] == -1 and children_left[node_number] == -1 and children_right[node_number] not in seen:  # if leaf and not earlier seen
            while node_number != 0:  # while we have not reached the root node
                parent_left = np.where(children_left == node_number)[0]  # check the parent left
                parent_right = np.where(children_right == node_number)[0]  # check the parent right
                if parent_left.size > 0:
                    parent_left = np.int(parent_left)
                    if feature[node_number] > -1:
                        countmatrix[feature[parent_left], feature[node_number]] += 1  # count
                    node_number = parent_left  # iterate until reach root node
                elif parent_right.size > 0:
                    parent_right = np.int(parent_right)
                    if feature[node_number] > -1:
                        countmatrix[feature[parent_right], feature[node_number]] += 1
                    node_number = parent_right
        seen.append(node_number)
    print(x)
    return countmatrix



def compare_feature_importance_bi(countmatrix_bi, rfc, linked_snp_unique, display_top=20):
    pathcount_top = np.sum(countmatrix_bi, axis=0).reshape((-1, 1))
    pathcount_top = sorted(zip(list(range(len(pathcount_top))), pathcount_top), key=lambda x: x[1] * -1)
    pathcount_top = np.asarray(pathcount_top[0:display_top])

    feature_importance_top = sorted(zip(list(range(len(rfc.feature_importances_))), rfc.feature_importances_),
                                    key=lambda x: x[1] * -1)
    feature_importance_top = np.asarray(feature_importance_top[0:display_top])

    comparison_feature_importance = np.append(pathcount_top[:, 0].reshape((-1, 1)),
                                              feature_importance_top[:, 0].reshape((-1, 1)), axis=1)
    comparison_feature_importance = pd.DataFrame(comparison_feature_importance,
                                                 columns=["pathcount", "feature importance"])

    score_pc_sim = set(comparison_feature_importance["pathcount"]) & set(linked_snp_unique)
    score_fi_sim = set(comparison_feature_importance["feature importance"]) & set(linked_snp_unique)

    return comparison_feature_importance, score_fi_sim, score_pc_sim


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

    return  scipy.sparse.csr_matrix.todense(countmatrix_bi) - chancenode * chancesamepath


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



if __name__ == '__main__':
    main()