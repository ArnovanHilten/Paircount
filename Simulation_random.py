import os
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
import create_toydata
import joblib
import glob
from sklearn import tree
import pandas as pd
import tree_analysis

# %% Create Toy Data
display_top = 20
print_trees = False
num_patients = 1000
num_features = 1000
num_experiment = 1000
num_trees = 1000
feature_frac = 0.5
linked_snp = [[[102, 1], [877, 2]], [[122, 2], [155, 1]], [[230, 1], [214, 2]], [[245, 1], [256, 1]]]
path = "/media/avanhilten/HDD1TB/simulations/noise_2_1000/"
path2 = "/media/avanhilten/HDD1TB/simulations/no_noise_2_1000/"
simulate = False
# %% Rewrite Linked SNP in handsome format

linked_snp_only = []
link_snp_element = []
linked_snp_unique = []
for linked in linked_snp:
    for element in linked:
        link_snp_element.append(element[0])
        linked_snp_unique.append(element[0])
    linked_snp_only.append(link_snp_element)
    link_snp_element = []
linked_snp_unique = np.unique(np.asarray(linked_snp_unique))

if simulate:
    t = 0

    while t < num_experiment:
        xtrain, ytrain = create_toydata.binomial_linked(num_patients, num_features, ind_linked=linked_snp,
                                                        random_seed=t)
        rfc = RandomForestClassifier(n_estimators=num_trees, max_features=feature_frac)

        ytrain = np.random.binomial(1, 0.3, num_patients)

        rfc.fit(xtrain, ytrain)
        countmatrix_uni, root_count = tree_analysis.get_pathcount(rfc, num_features)
        countmatrix_uni = scipy.sparse.csr_matrix(countmatrix_uni)
        if np.sum(countmatrix_uni) > 0:
            np.save(path + "sim" + str(t) + ".npy", countmatrix_uni)
            print((t, "with total", np.sum(countmatrix_uni)))
            t = t + 1

    t = 0

    while t < num_experiment:
        xtrain, ytrain = create_toydata.binomial_linked(num_patients, num_features, ind_linked=linked_snp,
                                                        random_seed=t)
        rfc = RandomForestClassifier(n_estimators=num_trees, max_features=feature_frac)

        rfc.fit(xtrain, ytrain)
        countmatrix_uni, root_count = tree_analysis.get_pathcount(rfc, num_features)
        countmatrix_uni = scipy.sparse.csr_matrix(countmatrix_uni)
        if np.sum(countmatrix_uni) > 0:
            np.save(path2 + "sim" + str(t) + ".npy", countmatrix_uni)
            np.save(path2 + "rn_" + str(t) + ".npy", root_count)
            joblib.dump(rfc, path2 + 'rfc' + str(t) + '.pkl')
            print((t, "with total", np.sum(countmatrix_uni)))
            t = t + 1

# %%

##
simfiles = glob.glob(path + "/sim*.npy")
total_sim = scipy.sparse.csr_matrix(np.zeros([num_patients, num_features]))
for simfile in simfiles:
    total_sim += scipy.load(simfile)[()]

total_sim_noise = total_sim / num_experiment


simfiles = glob.glob(path2 + "/sim*.npy")
total_sim = scipy.sparse.csr_matrix(np.zeros([num_patients, num_features]))
for simfile in simfiles:
    total_sim += scipy.load(simfile)[()]

total_sim_nonoise = total_sim / num_experiment


rootfiles = glob.glob(path2 + "/rn*.npy")
total_rn = np.zeros([len(rootfiles),num_trees ])
x = 0
for rootfile in rootfiles:
    total_rn[x,:] = np.load(rootfile)
simfiles = glob.glob(path2 + "/*.npy")


# rootfiles = glob.glob(path2 + "/rfc*.pkl")
# total_rn = np.zeros([len(rootfiles),num_trees ])
# x = 0
# for rootfile in rootfiles:
#     total_rn[x,:] = joblib.load(rootfile)
# simfiles = glob.glob(path2 + "/*.npy")
#


#%%  Correct for random chance
corrected_matrix_sim = np.array(total_sim_nonoise - scipy.sparse.csr_matrix.todense(total_sim_noise))
corrected_matrix_bi_sim = corrected_matrix_sim + np.transpose(corrected_matrix_sim)
corrected_matrix_bi_sim[np.tril_indices(corrected_matrix_bi_sim.shape[0], -1)] = 0
corrected_matrix_bi_sim.astype(int)


bt0_sim_w_uni = np.where(corrected_matrix_sim > 0)
bt0_sim_uni = corrected_matrix_sim[bt0_sim_w_uni]
pc_uni_v = np.reshape(np.array(bt0_sim_uni), [1, -1])
pc_uni_cordn = np.array(bt0_sim_w_uni)
pc_uni = np.transpose(np.concatenate((pc_uni_v, pc_uni_cordn)))
pc_uni_pd = pd.DataFrame(pc_uni, columns=['cvalue','x','y'])
pc_uni_pd = pc_uni_pd.sort_values(['cvalue'], ascending=False)
# corrected_matrix_bi_formula = tree_analysis.correct_countmatrix_bi_formula(total_sim_nonoise, root_count, rfc)
# corrected_matrix_bi_formula_p = tree_analysis.correct_countmatrix_bi_formula_precise(total_sim_nonoise, root_count, rfc)


bt0_sim = corrected_matrix_bi_sim[corrected_matrix_bi_sim > 0]
bt0_sim_w = np.where(corrected_matrix_bi_sim > 0)
# bt0_for = corrected_matrix_bi_formula[corrected_matrix_bi_formula > 0]
# bt0_for_w = np.where(corrected_matrix_bi_formula > 0)
# bt0_for_p = corrected_matrix_bi_formula_p[corrected_matrix_bi_formula_p > 0]
# bt0_for_p_w = np.where(corrected_matrix_bi_formula_p > 0)



# %%

# comparison_fi_sim, score_fi_sim, score_pi_sim = tree_analysis.compare_feature_importance_bi(corrected_matrix_bi_sim, rfc, linked_snp_unique, display_top=20, )
# comparison_fi_for, score_for_p_fi, score_for_p_pc = tree_analysis.compare_feature_importance_bi(corrected_matrix_bi_formula, rfc, linked_snp_unique, display_top=20, )
# comparison_fi_for_p, score_for_fi, score_for_pc = tree_analysis.compare_feature_importance_bi(corrected_matrix_bi_formula_p,rfc, linked_snp_unique, display_top=20, )

print_trees = True
i_tree = 0
if print_trees:
    for tree_in_forest in rfc.estimators_:
        with open('models/tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file=my_file)
        i_tree = i_tree + 1
    os.system("dot -Tpng models/tree_22.dot -o models/tree22.png")

