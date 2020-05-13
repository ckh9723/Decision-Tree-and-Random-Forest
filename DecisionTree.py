# Python modules
import random
import math
import sys
import pandas as pd
import numpy as np

# Self-defined modules
from Question import Question
from Nodes import Node, Leaf
import UtilityFunctions as uf


# ============================================================ Important Global Variables ========================================================================= #

# Dataset details
ds = pd.read_excel(r'Toy_Dataset.xlsx')  # Label MUST be at LAST column
feature_list = list(ds.columns)[0:-1]
label_list = list(set(ds['Label'][row] for row in range(len(ds.index))))
categorical_vals = {}  # A dictionary contains all categorical feature values (refer for-loop below)

for feature in feature_list:     # e.g. categorical_vals['Color'] = ['Red', 'Blue', 'Green']
    if isinstance(ds[feature][0],str):
        categorical_vals[feature] = list(set([ds[feature][row] for row in range(len(ds[feature].index))]))


# Variables to keep track of chosen features and values
features = []  # Feature space containing available features
chosen_vals = {f:[] for f in feature_list}   # Keep track of chosen values for CONTINUOUS feature e.g. {'length': [15,23,30]}
chosen_f = []   # Keep track of chosen CATEGORICAL feature e.g. chosen_f = ['Color', 'isRaining']


# ===================================================================== Methods ==================================================================================== #

def buildTree(ds, n_features=None, min_sample_split=-1):
    global features
    global chosen_f
    global chosen_vals

    # UPDATE FEATURE SPACE (Remove chosen categorical features)
    features = feature_list.copy()
    for f in chosen_f:
        if f in features:
            features.remove(f)  # Remove already chosen categorical features from feature space


    # SETUP HYPERPARAMETER CONDITIONS (Random Forest ONLY, unused if only building decision tree)
    if n_features is not None:  # Setup feature space based on 'n_features' specified
        if n_features > len(features):
            n_features = len(features)
        features = random.sample(population=features,k=n_features)  # Randomly sample feature to create feature space


    # FIND BEST SPLIT BY GAIN RATIO
    global_ent = computeEntropy(ds)
    gain_ratio, question = findBestSplit(ds,global_ent,min_sample_split)

    # If split not possible return leaf node; otherwise continue to split true & false
    if gain_ratio == 0 or gain_ratio == -1:  # 0 - No effective split can be made; -1 - Only 1 type of label is left
        return Leaf(ds=ds,labels=label_list)
    else:
        branches = []
        if isinstance(question.f_val,list):  # If feature is CATEGORICAL
            chosen_f.append(question.f)
            node_type = 'C'
            for f_val in question.f_val:
                current_q = Question(question.f, f_val)
                current_split_ds = current_q.split(ds)[0]
                if len(current_split_ds) == 0:
                    branches.append(Leaf(ds=ds,labels=label_list))
                else:
                    branches.append(buildTree(current_split_ds, n_features))
        else:  # If feature is CONTINUOUS
            chosen_vals[question.f].append(question.f_val)
            node_type = 'N'
            ds1,ds2 = question.split(ds)
            branches.append(buildTree(ds1, n_features))
            branches.append(buildTree(ds2, n_features))

        return Node(question,branches,node_type)


# Called by buildTree()
def findBestSplit(ds, global_ent, min_sample_split=-1):   # Find the best split for the current node of the tree
    global features
    global chosen_vals
    global categorical_vals

    best_gain_ratio = -1
    best_question = None
    label_num = len( set( ds['Label'][row] for row in range(len(ds.index)) ) )
    ds_splits = []  # Contains all split_ds to be fed to computeGainRatio()

    if len(ds) >= min_sample_split:
        if label_num != 0 or label_num != 1:
            for f in features:
                f_vals = list( dict.fromkeys( [ds[f][row] for row in range(len(ds[f].index))] ) )    # Get all unique values of the feature

                if isinstance(f_vals[0], np.int64) or isinstance(f_vals[0], np.float64):  # If feature is CONTINUOUS
                    for f_val in f_vals:
                        if f_val in chosen_vals[f]:  # Avoid repeating chosen feature value
                            continue
                        else:
                            q = Question(f,f_val)
                            ds1,ds2 = q.split(ds)
                            ds_splits.append(ds1)
                            ds_splits.append(ds2)
                            gain_ratio = computeGainRatio(ds_splits,global_ent)
                            ds_splits = []  # Empty the list after use

                            if gain_ratio > best_gain_ratio:
                                best_gain_ratio = gain_ratio
                                best_question = q

                else:  # If feature is CATEGORICAL
                    f_vals = categorical_vals[f]
                    for f_val in f_vals:
                        current_q = Question(f,f_val)
                        ds_splits.append(current_q.split(ds)[0])

                    gain_ratio = computeGainRatio(ds_splits,global_ent)
                    ds_splits = []  # Empty the list after use

                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_question = Question(f,f_vals)

    return best_gain_ratio,best_question


# Called by findBestSplit()
def computeGainRatio(ds_splits,global_ent):
    ent_list = []
    len_total = 0
    for ds in ds_splits:
        ent_list.append(computeEntropy(ds))
        len_total += len(ds)

    gain = global_ent
    for i in range(len(ds_splits)):
        gain -= ((len(ds_splits[i])/len_total) * ent_list[i])

    split_info = 0
    for ds in ds_splits:
        if len(ds) != 0:
            split_info += -(len(ds) / len_total) * math.log(len(ds) / len_total, 2)

    if split_info != 0:
        gain_ratio = gain / split_info
    else:
        if gain == 0:
            gain_ratio = 0
        else:
            gain_ratio = sys.float_info.max

    return gain_ratio


# Called by computeGainRatio()
def computeEntropy(ds):
    count_list = []   # A list containing the count for each label in the dataset
    for l in label_list:
        count_l = len(ds[ds.Label == l].index)
        count_list.append(count_l)
    total = len(ds.index)

    entropy = 0
    for i in range(len(count_list)):
        if count_list[i] != 0:
            entropy += -(count_list[i]/total)*math.log(count_list[i]/total,2)

    return entropy


# Get classification predictions the supplied test-set
def classify(node, test_ds):
    predictions = []
    for _,row in test_ds.iterrows():
        predictions.append(performClassification(node, row))
    return predictions

# Recursive function to traverse the tree to perform classification
def performClassification(node, row):
    if isinstance(node,Leaf):
        return node.prediction

    if node.type == 'N':    # If node type is 'Continuous' (N-Numerical)
        if node.question.match(row):
            return performClassification(node.branches[0], row)
        else:
            return performClassification(node.branches[1], row)

    else:   # If node type is 'Categorical', loop through every possible branch
        branch_n = len(node.question.f_val)
        for j in range(0, branch_n):
            cur_cat_val = node.question.f_val[j]    # Current Category Value
            if node.question.match(row,cur_cat_val):
                return performClassification(node.branches[j], row)



def main():
    # Split train/test
    train_ds, test_ds = uf.splitTrainTest(ds)

    # Build DT model and evaluate predictions on test-set
    tree = buildTree(train_ds)
    predictions = classify(tree,test_ds)
    accuracy = uf.evaluate(predictions,test_ds)
    uf.printOutput(train_ds=train_ds, model=tree, test_ds=test_ds, predictions=predictions, accuracy=accuracy)

    # Save model as 'mytree.txt' in current directory
    uf.saveModel(tree,'mytree.txt')

    # Load model from 'mytree.txt'
    loaded_tree = uf.loadModel('mytree.txt')


if __name__ == '__main__':
    main()