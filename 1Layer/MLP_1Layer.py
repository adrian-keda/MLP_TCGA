import numpy as np
import scipy.stats as sts
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

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
# Model name
name = 'MLP_1L'
# Preselected genes
pre_selected_gene = pd.read_csv(snakemake.input['permutations'], sep='\t', compression='gzip',index_col=0)
# Number of second repeats
second_repeat = 5
chuck = 80   # To split the 20k permutations in 80 chunks of 250 selections
chucknumber = int(snakemake.params['chunk']) # range(0 to N chuncks - 1)
permutation_range = np.arange(0, len(pre_selected_gene), int(len(pre_selected_gene)/chuck))
permutation_range_finish = np.arange(250,len(pre_selected_gene) + 250,int(len(pre_selected_gene)/chuck))
start_point = [i for k,i in enumerate(permutation_range)][chucknumber]
end_point = [i for k,i in enumerate(permutation_range_finish)][chucknumber]
basepath = './output_1Layer'


### INPUT DATA
# NA imputation
data = pd.read_csv(snakemake.input['inputfile'], sep='\t', index_col=0)
data['avg_CPG'] =  data['avg_CPG'].fillna(np.max(data['avg_CPG']))
data['avg_DNA_repair'] =  data['avg_DNA_repair'].fillna(np.max(data['avg_DNA_repair']))
data['avg_PIK_mTOR'] =  data['avg_PIK_mTOR'].fillna(np.max(data['avg_PIK_mTOR']))
data['avg_Cell_Cycle'] =  data['avg_Cell_Cycle'].fillna(np.max(data['avg_Cell_Cycle']))
data['avg_ALL'] =  data['avg_ALL'].fillna(np.max(data['avg_ALL']))
data = data.fillna(0)
# Feature selection
data = data[['case_con_zvalues', 'two_hit_zvalues', 'RDGV_by_nonRDGV', 'Tau_score',
       'oe_lof_upper', 'PTM_pvalue', 'PTV_frequency', 'amplification.freq',
       'deletion.freq', 'clin_zvalue', 'avg_ALL',
       'Bidirectional_Deletion', 'Edition_efficiency', 'Guide_Abundance',
       'Insertion', 'Insertion_Deletion', 'PAM_Proximal_Deletion', 'Is_positive']]
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


### DEFINING THE SEARCH SPACE
np.random.seed(35)
MLP = MLPClassifier(early_stopping = True, random_state = 35)
param_grid = {'hidden_layer_sizes' : [(8,), (16,), (32,), (64,)],
              'activation' : ['relu', 'logistic', 'tanh', 'identity'],
              'solver' : ['lbfgs', 'adam', 'sgd'],
              'alpha' : np.logspace(-3, 0, 4),
              'learning_rate' : ['constant', 'invscaling', 'adaptive'],
              'learning_rate_init' : [0.001, 0.01, 0.1],
              'max_iter' : [200, 500]
}


### UPDATED DATASET
data3 = data_for_training1.merge(data_y, left_index=True, right_index=True)
data3['Is_positive'] = data3['Is_positive'].apply(lambda x: 1 if x =='pos' else 0)
data3= data3[~data3.index.isin(['TTN'])]
cpg_list= data3[data3['Is_positive']==1].index.values
other_list = data3[data3['Is_positive']!=1].index.values


### LE FUNCTION
def process_iteration(i,pre_selcted_genes=pre_selected_gene):
    selected = [po for po in list(pre_selcted_genes.iloc[i].values)[0].split(',')]
    DF_gene = []
    DF_score = []
    GS_results = []
    data4 = data3.loc[selected].reset_index()
    # dataset division
    X = data4[x_col]
    y= data4[['Is_positive']]

    sss= StratifiedShuffleSplit(n_splits = second_repeat, test_size=0.2)
    j = 0
    KKKK = []
    M2 = []
    for train,test in sss.split(X[X.columns[1:]],y['Is_positive']) :
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        # MODEL Fitting
        grid_search = GridSearchCV(MLP, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train.values.ravel())
        best_model = grid_search.best_estimator_
        GS_results.append(pd.DataFrame([grid_search.best_params_], index=['%s_%s'%(i,j)]))
        # AUC
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]       
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label = 1)
        roc_auc = auc(fpr, tpr)
        auc_df = pd.DataFrame(data={'FPR':fpr, 'TPR':tpr, 'iters':'%s_%s'%(i,j),})
        M2.append(auc_df)        
        # PREDICTION and SCORE
        ML_pred = best_model.predict(X_test)
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
    KKKK.append([DF_gene, DF_score, M2, GS_results])
    return KKKK       
     

np.random.seed(35)
result =  Parallel(n_jobs=-1)(delayed(process_iteration)(i) for i in range(start_point, end_point) )

### EXTRACTING THE RESULTS
re_result_shape =np.array(result,dtype='object').reshape(-1, np.array(result,dtype='object').shape[-1])
GENE = pd.concat( [pd.concat(d) for d in re_result_shape[0::4]])
SCORE = pd.concat( [pd.concat(d) for d in re_result_shape[1::4]])
AUC_val= pd.concat( [pd.concat(d) for d in re_result_shape[2::4]])
GRID_SEARCH_RESULTS = pd.concat( [pd.concat(d) for d in re_result_shape[3::4]])
GENE2 = GENE


FIN_GENE = GENE2.groupby('Gene').sum(numeric_only=True)
FIN_GENE['EVENT'] = GENE2.groupby('Gene').count().values[:,1]
FIN_GENE['Likelyhood_CPG'] = FIN_GENE['ML_label'] / GENE2.groupby('Gene').count().values[:,1]


FIN_GENE.to_csv('%s/%s.ALL.permutation.%s.tsv'  %(basepath, name, chucknumber), sep='\t')
SCORE.to_csv('%s/%s.ALL.score.%s.tsv' %(basepath, name, chucknumber) , sep='\t')
AUC_val.to_csv('%s/%s.ALL.AUC.%s.tsv' %(basepath, name, chucknumber) , sep='\t')
GRID_SEARCH_RESULTS.to_csv('%s/%s.ALL.GS_params.%s.tsv' %(basepath, name, chucknumber) , sep='\t')