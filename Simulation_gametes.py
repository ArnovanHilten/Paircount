#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tree_analysis
import sklearn.linear_model
import glob
#  Create Toy Data


num_trees = 100
feature_frac = 0.8


path = "/home/avanhilten/Gametes/repro_44procent25snp_dataset_EDM-2/"
files = list(glob.glob(path +'*.txt'))
njobs = 14
rf_acc = np.zeros(len(files))
lr_acc = np.zeros(len(files))
top_uni = np.zeros((len(files), 3))
top_bi = np.zeros((len(files), 3))
top_fi = np.zeros((len(files), 3))
for file_num, path in enumerate(files):
#%%
    print(file_num)
    gametedata = pd.read_table(path)
    traindata = gametedata.sample(frac=0.5)
    valdata = gametedata.loc[~gametedata.index.isin(traindata.index)]
    xtrain = traindata.loc[:, traindata.columns != 'Class']
    ytrain = np.ravel(traindata[['Class']])
    rfc = RandomForestClassifier(n_estimators=num_trees, max_features=feature_frac, n_jobs=njobs)


    rfc.fit(xtrain, ytrain)
    countmatrix_uni = tree_analysis.get_pathcount_parentonly(rfc, xtrain.shape[1], verbose=0)

    countmatrix_bi = countmatrix_uni + np.transpose(countmatrix_uni)
    countmatrix_bi[np.tril_indices(len(countmatrix_bi), -1)] = 0
    countmatrix_bi.astype(int)

    bt0_sim_w_uni = np.where(countmatrix_uni)
    bt0_sim_uni = countmatrix_uni[bt0_sim_w_uni]
    pc_uni_v = np.reshape(np.array(bt0_sim_uni), [1, -1])
    pc_uni_cordn = np.array(bt0_sim_w_uni)
    pc_uni = np.array(np.transpose(np.concatenate((pc_uni_v, pc_uni_cordn))), dtype=int)
    pc_uni_pd = pd.DataFrame(pc_uni, columns=['cvalue', 'x', 'y'])
    pc_uni_pd = pc_uni_pd.sort_values(['cvalue'], ascending=False)

    bt0_sim_w_bi = np.where(countmatrix_bi)
    bt0_sim_bi = countmatrix_bi[bt0_sim_w_bi]
    pc_bi_v = np.reshape(np.array(bt0_sim_bi), [1, -1])
    pc_bi_cordn = np.array(bt0_sim_w_bi)
    pc_bi = np.array(np.transpose(np.concatenate((pc_bi_v, pc_bi_cordn))), dtype=int)
    pc_bi_pd = pd.DataFrame(pc_bi, columns=['cvalue', 'x', 'y'])
    pc_bi_pd = pc_bi_pd.sort_values(['cvalue'], ascending=False)

    FI_max = np.argmax(rfc.feature_importances_)

    feature_importance = sorted(zip(list(range(len(rfc.feature_importances_))), rfc.feature_importances_),
                                key=lambda x: x[1] * -1)
    feature_importance_pc  = sorted(zip(list(range(len(rfc.feature_importances_))), np.sum(countmatrix_uni + np.transpose(countmatrix_uni),axis=1)),
                                key=lambda x: x[1] * -1)
    top_fi[file_num,0] = feature_importance[0][0]
    top_fi[file_num, 1] = feature_importance[0][1]
    xval = valdata.loc[:, valdata.columns != 'Class']
    yval = np.ravel(valdata[['Class']])

    pval = rfc.predict(xval)
    rf_acc[file_num] = sklearn.metrics.accuracy_score(yval,pval)
    print(" rf accuracy" + str(rf_acc[file_num]))

    lr1 = sklearn.linear_model.LogisticRegression()
    lr1.fit(xtrain, ytrain)
    #%%

    pval_lr = lr1.predict(xval)
    rsquared_lr_train = lr1.score(xtrain, ytrain)
    rsquared_lr_val = lr1.score(xval, yval)
    lr_acc[file_num] = sklearn.metrics.accuracy_score(yval,pval_lr)
    print(" lr accuracy" + str(lr_acc[file_num]))

    top_uni[file_num, 0] = xtrain.shape[1]-2in set(pc_uni_pd.iloc[0].tolist()[1:3]) or  xtrain.shape[1]-1 in set(pc_uni_pd.iloc[0].tolist()[1:3])
    top_uni[file_num, 1] = pc_uni_pd.iloc[0].tolist()[0]
    top_uni[file_num, 2] = pc_uni_pd.iloc[1].tolist()[0]

    top_bi[file_num, 0] = xtrain.shape[1]-2 in set(pc_bi_pd.iloc[0].tolist()[1:3]) or  xtrain.shape[1]-1 in set(pc_bi_pd.iloc[0].tolist()[1:3])
    top_bi[file_num, 1] = pc_bi_pd.iloc[0].tolist()[0]
    top_bi[file_num, 2] = pc_bi_pd.iloc[1].tolist()[0]

np.savez(path + "experiment.npz",lr_acc,rf_acc,top_uni,top_bi)
#%%
print(np.sum(top_uni[:,0]))
print(np.sum(top_bi[:,0]))
print(sum(rf_acc >0.5))
print(sum(lr_acc >0.5))
print(sum(rf_acc > lr_acc))
totalcm = np.sum(np.sum(countmatrix_uni))
print(totalcm)
