"""
Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import data_import as d
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_importance
import os
from sklearn.model_selection import train_test_split
label_encoder = LabelEncoder()
import itertools
from sklearn.model_selection import train_test_split, StratifiedKFold


# =============================================================================
# data input
# =============================================================================
data_input = d.data_input


# =============================================================================
# label
# =============================================================================
data_label = d.data_label
data_label = label_encoder.fit_transform(data_label)
classes = np.unique(data_label)
n_classes = classes.shape[0]

######################################################################

positive = total = np.sum(data_label)
negative = len(data_label)-positive
metric_final = 0

# =============================================================================
# hyperparameter search
# =============================================================================

cv_splits_top = 10
for x in np.arange(8, cv_splits_top, 1):
    cv_splits = x
    
    # =============================================================================
    # XGBoost
    # =============================================================================
    print('XGBOOST: ')
    
    # =============================================================================
    #  loss weighting
    # =============================================================================
    weight = negative/positive
    
    # =============================================================================
    #  define model
    # =============================================================================
    # model = XGBClassifier(scale_pos_weight=weight, n_jobs=3)
    
    
    # =============================================================================
    # cross validation 
    # =============================================================================
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True)
    skf.get_n_splits(data_input, data_label)
    final_val_acc = list()
            
    i = 0
    val0_0, val1_0 = list(), list()
    val0_sum, val1_sum, = [0]*1000, [0]*1000
    xgb_auc, acc, f1, precision, recall, specificity = 0,0,0,0,0,0
    
    for train_index, val_index in skf.split(data_input, data_label):
        trainX, testX = data_input[train_index], data_input[val_index]
        trainy, testy = data_label[train_index], data_label[val_index]
     
    
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(testy))]
        # fit a model
        model = XGBClassifier(scale_pos_weight=weight)
        
        # =============================================================================
        # loss curves
        # =============================================================================
        eval_set = [(trainX, trainy), (testX, testy)]
        model.fit(trainX, trainy, eval_metric='logloss', early_stopping_rounds=10, eval_set = eval_set, verbose=True)
        results = model.evals_result()
        
        val0_0 = results['validation_0']['logloss']
        val0_sum = [(x + y) for (x,y) in zip(val0_sum, val0_0)]
        val1_0 = results['validation_1']['logloss']
        val1_sum = [(x + y) for (x,y) in zip(val1_sum, val1_0)] 
        
        ### AUC
        ns_fpr_list = [0] *2
        ns_tpr_list = [0] *2
        xgb_fpr_list = [0] *32
        xgb_tpr_list = [0] *32
        
        # predict probabilities
        xgb_probs = model.predict_proba(testX)
        
        # keep probabilities for the positive outcome only
        xgb_probs = xgb_probs[:, 1]
        test_prediction = np.where(xgb_probs>=0.5, 1, 0)
        
        # calculate scores
        ns_auc = roc_auc_score(testy, ns_probs)
        xgb_auc += roc_auc_score(testy, xgb_probs)
        
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
        xgb_fpr, xgb_tpr, _ = roc_curve(testy, xgb_probs)
        
    
        # =============================================================================
        # metrics for X classes
        # =============================================================================
        
        ### accuracy = (tp + tn) / /(tp + tn + fp + fn)
        acc += accuracy_score(test_prediction, testy)
        # print('accuracy: ', round(acc, 2)/i)
        
        # =============================================================================
        # metrics for 2 classes
        # =============================================================================
        
        ## confusion matrix
        # fig, ax = plt.subplots(figsize=(10,10))
        cm = confusion_matrix(test_prediction, testy)
        # cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        # plt.title('XGBOOST')
        # plt.savefig(os.path.join(os.getcwd(), 'results/Confusion Matrix'), dpi=400)
        tn = cm[0,0]
        tp = cm[1,1]
        fp = cm[0,1]
        fn = cm[1,0]
    
        ## f1_score = 2 * (pre * rec) / (pre + rec)
        f1 += f1_score(test_prediction, testy)
        # print('f1/ Dice : ', round(f1, 2)/i)
        
        # precision_score = tp / (tp + fp)
        precision += precision_score(test_prediction, testy)
        # print('precision: ', round(precision, 2)/i)
        
        ### recall_score = tp / (tp + fn)
        recall += recall_score(test_prediction, testy)
        # print('recall/ sensitivity: ', round(recall, 2)/i)
        
        ### specificity = tn/(tn + fp)
        specificity += tn/(tn + fp)
        # print('specificity : ', round(specificity, 2)/i)
    
        i +=1
        
        
    ### accuracy = (tp + tn) / /(tp + tn + fp + fn)
    acc = acc/cv_splits

    # summarize scores
    xgb_auc = xgb_auc/cv_splits
    
    ## f1_score = 2 * (pre * rec) / (pre + rec)
    f1 = f1/cv_splits
    
    # precision_score = tp / (tp + fp)
    precision = precision/cv_splits
    
    ### recall_score = tp / (tp + fn)
    recall = recall/cv_splits
    
    ### specificity = tn/(tn + fp)
    specificity = specificity/cv_splits
    
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::', cv_splits)
    
    if metric_final < xgb_auc:
        metric_final = xgb_auc
        acc_final = acc
        xgb_auc_final = xgb_auc
        f1_final = f1
        precision_final = precision
        recall_final = recall
        specificity_final = specificity
        cv_splits_final = cv_splits

        val0_final, val1_final = [], []
        model_final = model
        for value in val0_sum:
            val0_final.append(value/cv_splits)
        for value in val1_sum:
            val1_final.append(value/cv_splits)




    
print('accuracy: ', round(acc_final, 2))
print('ROC AUC:',  round(xgb_auc_final, 2))
print('f1/ Dice : ', round(f1_final, 2))
print('precision: ', round(precision_final, 2))
print('recall/ sensitivity: ', round(recall_final, 2))
print('specificity : ', round(specificity_final, 5))
print('cv_splits_final: ', cv_splits_final)


epochs = len(val0_final)
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, val0_final, color='r')
ax.plot(x_axis, val1_final, color='b')
ax.legend(('train', 'test'))
plt.ylabel('Classification Error')
plt.xlabel('Epochs')
plt.savefig(os.path.join(os.getcwd(), 'results/loss'), dpi=400)


# =============================================================================
#  plot feature importance
# =============================================================================
plt.figure(figsize=(10,10))
plot_importance(model_final, max_num_features=5, title='Feature Importance Top 5', color='b')
plt.savefig(os.path.join(os.getcwd(), 'results/feature_importance'), dpi=400)
plt.show()



# plot the roc curve for the model
plt.figure(figsize=(10,10))
plt.plot(ns_fpr, ns_tpr, linestyle='--', color='r')
plt.plot(xgb_fpr, xgb_tpr, marker='.', label='XGBoost', color='b')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title ('AUC ROC')
plt.savefig(os.path.join(os.getcwd(), 'results/ROC AUC'), dpi=400)
plt.show()



print('--------------')