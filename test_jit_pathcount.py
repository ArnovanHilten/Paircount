import os
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
import create_toydata
import Paircount
from time import time
# %% Create Toy Data
display_top = 20
print_trees = False
num_patients = 1000
num_features = 1000
num_experiment = 2
num_trees = 1000
feature_frac = 0.5
linked_snp = [[[102, 1], [877, 2]], [[122, 2], [155, 1]], [[230, 1], [214, 2]], [[245, 1], [256, 1]]]
path = "/media/avanhilten/pHDD1TB/Simulations/paircount/simulation_random/1_noise/"
path2 = "/media/avanhilten/pHDD1TB/Simulations/paircount/simulation_random/1_no_noise"
simulate = True

xtrain, ytrain = create_toydata.binomial_linked(num_patients, num_features, ind_linked=linked_snp,
                                                random_seed=1)
rfc = RandomForestClassifier(n_estimators=num_trees, max_features=feature_frac)

rfc.fit(xtrain, ytrain)

print('Random forest trained')
print("Start analyzing")
t = time()
countmatrix_pool = Paircount.get_pathcount_forest_pool(rfc, num_features)
countmatrix_pool = scipy.sparse.csr_matrix(countmatrix_pool)
print('Map pool - done in {} s'.format(time() - t))

t = time()
countmatrix_async = Paircount.get_pathcount_forest_async(rfc, num_features)
countmatrix_async = scipy.sparse.csr_matrix(countmatrix_async)
print('asyncl - done in {} s'.format(time() - t))

t = time()
countmatrix_single = Paircount.get_pathcount_forest_single(rfc, num_features)
countmatrix_single = scipy.sparse.csr_matrix(countmatrix_single)
print('Single - done in {} s'.format(time() - t))

print(countmatrix_single.sum())
print(countmatrix_single.sum())
print(countmatrix_pool.sum())