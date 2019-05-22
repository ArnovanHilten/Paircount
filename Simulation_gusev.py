
import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor
import create_toydata
import joblib
import glob
import tree_analysis
import matplotlib.pyplot as plt

# %% Create Toy Data
print("start OUTDATED check NONOISE")
display_top = 20
print_trees = False
num_patients = 4000
num_features = 1000
num_experiment = 10
num_trees = 100
feature_frac = 0.5
linked_snp = [[[102, 1], [877, 2]], [[122, 2], [155, 1]], [[230, 1], [214, 2]], [[245, 1], [256, 1]]]
path = "/media/avanhilten/pHDD1TB/Simulations/noise/"
path2 = "/media/avanhilten/pHDD1TB/Simulations/no_noise/"
simulate = True
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
        xtrain, ytrain = create_toydata.gusev_linked(num_patients, num_features, ind_linked=linked_snp,
                                                        random_seed=t)
        rfc = RandomForestRegressor(n_estimators=num_trees, max_features=feature_frac)

        ytrain = np.random.randn(num_patients)

        rfc.fit(xtrain, ytrain)
        countmatrix_uni = tree_analysis.get_pathcount_parentonly(rfc, num_features)
        countmatrix_uni = scipy.sparse.csr_matrix(countmatrix_uni)
        if np.sum(countmatrix_uni) > 0:
            np.save(path + "sim" + str(t) + ".npy", countmatrix_uni)
            print((t, "with total", np.sum(countmatrix_uni)))
            t = t + 1

    t = 0

    while t < num_experiment:
        xtrain, ytrain = create_toydata.gusev_linked(num_patients, num_features, ind_linked=linked_snp,
                                                        random_seed=t)
        rfc = RandomForestRegressor(n_estimators=num_trees, max_features=feature_frac)

        rfc.fit(xtrain, ytrain)
        countmatrix_uni = tree_analysis.get_pathcount_parentonly(rfc, num_features)
        countmatrix_uni = scipy.sparse.csr_matrix(countmatrix_uni)
        if np.sum(countmatrix_uni) > 0:
            np.save(path2 + "sim" + str(t) + ".npy", countmatrix_uni)
            joblib.dump(rfc, path2 + 'rfc' + str(t) + '.pkl')
            print((t, "with total", np.sum(countmatrix_uni)))
            t = t + 1

# %%

##
simfiles = glob.glob(path + "/sim*.npy")
total_sim = scipy.sparse.csr_matrix(np.zeros([num_features, num_features]))
for simfile in simfiles:
    total_sim += scipy.load(simfile)[()]

total_sim_noise = total_sim / num_experiment


simfiles = glob.glob(path2 + "/sim*.npy")
total_sim = scipy.sparse.csr_matrix(np.zeros([num_features, num_features]))
for simfile in simfiles:
    total_sim += scipy.load(simfile)[()]

total_sim_nonoise = total_sim / num_experiment

cm_noise = scipy.sparse.csr_matrix.todense(total_sim_noise)
cm_noise = cm_noise + np.transpose(cm_noise)
cm_noise[np.tril_indices(cm_noise.shape[0], -1)] = 0
cm_noise = np.sum(cm_noise, axis=1)

cm_sim = scipy.sparse.csr_matrix.todense(total_sim)
cm_sim = cm_sim + np.transpose(cm_sim)
cm_sim[np.tril_indices(cm_sim.shape[0], -1)] = 0
cm_sim = np.sum(cm_sim, axis=1)


print('start plot')
plt.figure()
plt.hist(cm_noise,bins = 100)
plt.title('Distribution simulation with noisy labels')
plt.ylabel('Paircount number')
plt.show()
plt.savefig(path + "noise_dist.png")
plt.close()



print('start plot')
plt.figure()
plt.hist(cm_sim,bins = 100)
plt.title('Distribution simulation with sim labels')
plt.ylabel('Paircount number')
plt.show()
plt.savefig(path + "sim_distr.png")
plt.close()




#
# #%%  Correct for random chance
# corrected_matrix_sim = np.array(total_sim_nonoise - scipy.sparse.csr_matrix.todense(total_sim_noise))
# corrected_matrix_bi_sim = corrected_matrix_sim + np.transpose(corrected_matrix_sim)
# corrected_matrix_bi_sim[np.tril_indices(corrected_matrix_bi_sim.shape[0], -1)] = 0
# corrected_matrix_bi_sim.astype(int)
#
#
# bt0_sim_w_uni = np.where(corrected_matrix_sim > 0)
# bt0_sim_uni = corrected_matrix_sim[bt0_sim_w_uni]
# pc_uni_v = np.reshape(np.array(bt0_sim_uni), [1, -1])
# pc_uni_cordn = np.array(bt0_sim_w_uni)
# pc_uni = np.transpose(np.concatenate((pc_uni_v, pc_uni_cordn)))
# pc_uni_pd = pd.DataFrame(pc_uni, columns=['cvalue','x','y'])
# pc_uni_pd = pc_uni_pd.sort_values(['cvalue'], ascending=False)
# # corrected_matrix_bi_formula = tree_analysis.correct_countmatrix_bi_formula(total_sim_nonoise, root_count, rfc)
# # corrected_matrix_bi_formula_p = tree_analysis.correct_countmatrix_bi_formula_precise(total_sim_nonoise, root_count, rfc)
#
#
# bt0_sim = corrected_matrix_bi_sim[corrected_matrix_bi_sim > 0]
# bt0_sim_w = np.where(corrected_matrix_bi_sim > 0)
# # bt0_for = corrected_matrix_bi_formula[corrected_matrix_bi_formula > 0]
# # bt0_for_w = np.where(corrected_matrix_bi_formula > 0)
# # bt0_for_p = corrected_matrix_bi_formula_p[corrected_matrix_bi_formula_p > 0]
# # bt0_for_p_w = np.where(corrected_matrix_bi_formula_p > 0)
#
#
#
# # %%
#
# # comparison_fi_sim, score_fi_sim, score_pi_sim = tree_analysis.compare_feature_importance_bi(corrected_matrix_bi_sim, rfc, linked_snp_unique, display_top=20, )
# # comparison_fi_for, score_for_p_fi, score_for_p_pc = tree_analysis.compare_feature_importance_bi(corrected_matrix_bi_formula, rfc, linked_snp_unique, display_top=20, )
# # comparison_fi_for_p, score_for_fi, score_for_pc = tree_analysis.compare_feature_importance_bi(corrected_matrix_bi_formula_p,rfc, linked_snp_unique, display_top=20, )
#
# print_trees = False
# i_tree = 0
# if print_trees:
#     for tree_in_forest in rfc.estimators_:
#         with open('models/tree_' + str(i_tree) + '.dot', 'w') as my_file:
#             my_file = tree.export_graphviz(tree_in_forest, out_file=my_file)
#         i_tree = i_tree + 1
#     os.system("dot -Tpng models/tree_22.dot -o models/tree22.png")
#
