import numpy as np
import scipy.stats as sts
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import sys


### PARAMETERS
first_permu = 20000
second_repeat = 5
basepath = './output_final_models'
name = 'MLP_1Layer'


### INPUT DATA
data = pd.read_csv('./TCGA.integrative.model.input.by.rdgv.noSOD11.tsv', 
                   sep='\t', index_col=0)


### PROCESING THE DATA
# NA imputation
data['avg_CPG'] =  data['avg_CPG'].fillna(np.max(data['avg_CPG']))
data['avg_DNA_repair'] =  data['avg_DNA_repair'].fillna(np.max(data['avg_DNA_repair']))
data['avg_PIK_mTOR'] =  data['avg_PIK_mTOR'].fillna(np.max(data['avg_PIK_mTOR']))
data['avg_Cell_Cycle'] =  data['avg_Cell_Cycle'].fillna(np.max(data['avg_Cell_Cycle']))
data['avg_ALL'] =  data['avg_ALL'].fillna(np.max(data['avg_ALL']))
data=data.fillna(0)
# Feature selection
data = data[['case_con_zvalues', 'two_hit_zvalues', 'RDGV_by_nonRDGV', 'Tau_score',
       'oe_lof_upper', 'PTM_pvalue', 'PTV_frequency', 'amplification.freq',
       'deletion.freq', 'clin_zvalue', 'avg_ALL',
       'Bidirectional_Deletion', 'Edition_efficiency', 'Guide_Abundance',
       'Insertion', 'Insertion_Deletion', 'PAM_Proximal_Deletion', 'Is_positive']]
np.random.seed(35)
x_col = ['case_con_zvalues', 'two_hit_zvalues', 'RDGV_by_nonRDGV', 'Tau_score',
       'oe_lof_upper', 'PTM_pvalue', 'PTV_frequency', 'amplification.freq',
       'deletion.freq', 'clin_zvalue', 'avg_ALL',
       'Bidirectional_Deletion', 'Edition_efficiency', 'Guide_Abundance',
       'Insertion', 'Insertion_Deletion', 'PAM_Proximal_Deletion', ]
data_y = data.pop('Is_positive')
data_x = data
# Normalizing the data
scaler = StandardScaler()
data_for_training1 = pd.DataFrame(data = scaler.fit_transform(data_x ) ,index=data_x.index, 
            columns=data_x.columns)


### UPDATED DATASET
data3 = data_for_training1.merge(data_y, left_index=True, right_index=True)
data3['Is_positive'] = data3['Is_positive'].apply(lambda x: 1 if x =='pos' else 0)
data3= data3[~data3.index.isin(['TTN'])]
cpg_list= data3[data3['Is_positive']==1].index.values
other_list = data3[data3['Is_positive']!=1].index.values


### LE FUNCTION
def process_iteration(i):
    DF_gene = []
    DF_score = []
    selected_gene_other = np.random.choice(other_list, len(cpg_list), replace=False)
    fin_gene = np.concatenate((cpg_list,selected_gene_other))
    data4= data3[data3.index.isin(fin_gene)].reset_index()
    # dataset division
    X = data4[x_col]
    y = data4[['Is_positive']]
    
    sss= StratifiedShuffleSplit(n_splits=second_repeat, test_size=0.2)
    j = 0
    KKKK = []
    M2 = []
    # repeat second_repeat times
    for train,test in sss.split(X[X.columns[1:]],y['Is_positive']):
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        # MODEL Fitting
        MLP = MLPClassifier(early_stopping = True, random_state = 35, hidden_layer_sizes = (1,), activation = 'logistic',
                            solver = 'lbfgs', learning_rate = 'constant', learning_rate_init = 0.001,
                            alpha = 1, max_iter = 200)
        MLP.fit(X_train, y_train.values.ravel())
        # AUC
        y_pred_proba = MLP.predict_proba(X_test)[:, 1] 
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)
        auc_df = pd.DataFrame(data={'FPR':fpr, 'TPR':tpr, 'iters':'%s_%s'%(i,j),})
        M2.append(auc_df)        
        # PREDICTION and SCORE
        ML_pred = MLP.predict(X_test)
        precision = precision_score(y_test, ML_pred)
        recall= recall_score(y_test, ML_pred)
        f1score = f1_score(y_test, ML_pred)
        accuracy =accuracy_score(y_test, ML_pred)
        # LIKELIHOOD TABLE
        daf = pd.DataFrame(data = {'Gene': data4[['Gene','Is_positive']].iloc[test].values[:,0],
                      'iters': np.repeat('%s_%s'%(i,j),len(test)),
                      'True_label': np.fromiter(data4[['Gene','Is_positive']].iloc[test].values[:,1],dtype=int),
                      'ML_label': np.fromiter(ML_pred,dtype=int) }
                          )
        # SCORES TABLE
        DDD = pd.DataFrame(data={'AUC':roc_auc, 'Precision':precision, "Recall":recall, 'F1_score':f1score,
                                 'ACCURACY':accuracy}, index=['%s_%s'%(i,j)] )
        DF_gene.append(daf)
        DF_score.append(DDD)

        j+=1
    KKKK.append([DF_gene,DF_score,M2])
    return KKKK   

np.random.seed(35)
result =  Parallel(n_jobs=-1)(delayed(process_iteration)(i) for i in range(first_permu))


### EXTRACTING THE RESULTS
re_result_shape =np.array(result,dtype='object').reshape(-1, np.array(result,dtype='object').shape[-1])
GENE = pd.concat( [pd.concat(d) for d in re_result_shape[0::3]])
SCORE = pd.concat( [pd.concat(d) for d in re_result_shape[1::3]])
AUC_val= pd.concat( [pd.concat(d) for d in re_result_shape[2::3]])
GENE2 = GENE


FIN_GENE = GENE2.groupby('Gene').sum(numeric_only=True)
FIN_GENE['EVENT'] = GENE2.groupby('Gene').count().values[:,1]
FIN_GENE['Likelyhood_CPG'] = FIN_GENE['ML_label'] / GENE2.groupby('Gene').count().values[:,1]


n_permut = first_permu * second_repeat
FIN_GENE.to_csv(f'{basepath}/{name}_{n_permut}.permutation.tsv', sep = '\t')
SCORE.to_csv(f'{basepath}/{name}_{n_permut}.SCORE.tsv', sep = '\t')
AUC_val.to_csv(f'{basepath}/{name}_{n_permut}.AUC_score.tsv', sep = '\t')