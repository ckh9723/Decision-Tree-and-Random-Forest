# Python modules
import pandas as pd
import numpy as np
import random

# Self-defined modules
from Nodes import Node,Leaf
from Question import Question


def splitTrainTest(ds):
    len_train = round(len(ds)*0.7)
    train_ds = pd.DataFrame()
    test_ds = ds.copy()
    label_list = list(set(ds['Label'][row] for row in range(len(ds.index))))

    for label in label_list:
        ds_label = ds[ds['Label'] == label].copy()
        frac_ds_label = len(ds_label)/len(ds)
        sample_len = round(frac_ds_label*len_train)
        ds_label_sample = ds_label.sample(n=sample_len).copy()
        train_ds = train_ds.append(ds_label_sample)
        test_ds.drop(ds_label_sample.index,inplace=True)
    train_ds.reset_index(drop=True,inplace=True)
    test_ds.reset_index(drop=True,inplace=True)

    if len(test_ds) == 0:
        test_ds = train_ds.copy()

    return train_ds,test_ds


# Only used in Random Forest
def createBootstrap(ds,n_trees):
    bootstrap_ds = {}   # Map bootstrap_ds to tree_index
    oob_dict = {}       # Map OOB rows to tree_index

    for n in range(n_trees):
        while True:
            bootstrap_indices = []
            oob_indices = [i for i in range(len(ds))]

            for i in range(len(ds)):
                selected_index = random.randint(0,len(ds)-1)
                bootstrap_indices.append(selected_index)
                if selected_index in oob_indices:
                    oob_indices.remove(selected_index)

            ds_bootstrap = ds.iloc[bootstrap_indices].copy()
            ds_bootstrap.reset_index(drop=True,inplace=True)
            bootstrap_ds[n] = ds_bootstrap
            for j in oob_indices:
                if j in oob_dict:
                    oob_dict[j].append(n)
                else:
                    oob_dict[j] = [n]

            if len(oob_indices) != 0:
                break

        oob_ds = pd.DataFrame([ds.iloc[i] for i in oob_dict].copy())
        oob_ds.reset_index(drop=True,inplace=True)

    return bootstrap_ds,oob_dict,oob_ds


# Find label with max probability & add to final_predictions  e.g. {'Apple': 0.8, 'Lemon': 0.2} => [Apple]
# If there is a tie, toss a coin  e.g. {'Apple': 0.5, 'Lemon': 0.5} => [Apple]
def get_PredictionLabel(prediction):
    prediction_label = None
    max = 0
    for label in prediction:
        if prediction[label] > max:
            prediction_label = label
            max = prediction[label]

        elif prediction[label] == max:
            if prediction[label] != 0:
                if random.randint(1,2) == 1:
                    prediction_label = label

    return prediction_label


def evaluate(predictions,test_ds):
    true_labels = [test_ds['Label'][row] for row in range(len(test_ds.index))]

    # Get label from predictions
    predictions_label = []
    for p in predictions:
        p_label = get_PredictionLabel(p)
        predictions_label.append(p_label)

    # Calculate accuracy
    sum = 0
    for i in range(len(test_ds)):
        if true_labels[i] in predictions_label[i]:
            sum += 1
    accuracy = (sum/len(true_labels))*100

    return accuracy


def saveModel(model,file_path):
    f = open(file_path,'w')

    if isinstance(model,list):
        f.write('{}\n'.format(len(model)))
        for tree in model:
            writeTree(tree,f)
            f.write('-----------------------------------\n')
    else:
        f.write('1\n')
        writeTree(model,f)
        f.write('-----------------------------------\n')
    f.close()

def writeTree(node,file):
    if isinstance(node,Leaf):
        prediction_dict = node.prediction
        file.write('{')
        for key in prediction_dict:
            file.write('\'{}\'{},'.format(key,prediction_dict[key]))
        file.write('}\n')

    else:
        node_type = node.type
        n_branch = len(node.branches)
        feature = node.question.f
        feature_val = node.question.f_val

        if node_type == 'N':
            file.write('{},{},\'{}\'{}\n'.format(node_type,n_branch,feature,feature_val))
        else:
            file.write('{},{},\'{}\''.format(node_type,n_branch,feature))
            for f_val in feature_val:
                file.write('\'{}\''.format(f_val))
            file.write('\n')

        for i in range(n_branch):
            writeTree(node.branches[i],file)


def loadModel(file_path):
    f = open(file_path)
    line_list = f.readlines()   # Read all lines
    f.close()

    length = int(line_list[0])
    line_list = line_list[1:]

    if length > 1:
        forest = []
        for i in range(length):
            line_list,tree = loadTree(line_list)
            forest.append(tree)
        return forest
    else:
        _, tree = loadTree(line_list)
        return tree


def loadTree(line_list,line_n=0):
    node_info = line_list[line_n]
    i = 0
    first_c = node_info[i]

    # If node is LEAF
    if first_c == '{':
        prediction_dict = {}
        i+=1

        while node_info[i] != '}':
            key = ''
            key_value = ''

            # Get key name
            i += 1
            while node_info[i] != '\'':
                key+=node_info[i]
                i+=1

            # Get key value
            i+=1
            while node_info[i] != ',':
                key_value+=node_info[i]
                i+=1
            key_value = np.float64(key_value)

            # Add key-value prediction pair
            prediction_dict[key] = key_value
            i+=1

        next_line = line_n+1
        if line_n == 0:  # return new_line_list after finish loading a tree (Remove loaded tree from line_list)
            next_tree_line = next_line+1
            new_line_list = line_list[next_tree_line:]
            return new_line_list,Leaf(prediction=prediction_dict)
        else:
            return next_line, Leaf(prediction=prediction_dict)

    # If node is NOT LEAF
    else:
        # Important variables
        node_type = ''
        n_branch = ''
        branches = []
        feature = ''
        feature_val = None  # Can be single-valued or a list depending on node_type

        # Get node type
        node_type = node_info[i]

        # Get number of branches
        i+=2
        while node_info[i] != ',':
            n_branch+=node_info[i]
            i+=1
        n_branch = int(n_branch)

        # Get feature name
        i+=2
        while node_info[i] != '\'':
            feature+=node_info[i]
            i+=1

        # Get feature value
        if node_type == 'N':
            feature_val = ''
            i+=1
            while node_info[i] != '\n':
                feature_val+=node_info[i]
                i+=1
            feature_val = np.float64(feature_val)
        else:
            feature_val = []
            for j in range(n_branch):
                i+=2
                cur_feature_val = ''

                while node_info[i] != '\'':
                    cur_feature_val+=node_info[i]
                    i+=1
                feature_val.append(cur_feature_val)

        # Recursive call to load the entire tree
        next_line = line_n + 1
        for i in range(n_branch):
            next_line,branch = loadTree(line_list,line_n=next_line)
            branches.append(branch)

        if line_n == 0:  # new_line_list after finish loading a tree (Remove loaded tree from line_list)
            next_tree_line = next_line+1
            new_line_list = line_list[next_tree_line:]
            return new_line_list, Node(Question(feature,feature_val),branches,node_type)
        else:
            return next_line, Node(Question(feature, feature_val), branches, node_type)


def printTree(node,spacing=''):
    # Base case: Reached the leaf node
    if isinstance(node,Leaf):

        # Format output to 2 decimal-places
        prediction = node.prediction
        for label in prediction:
            prediction[label] = round(prediction[label],2)

        print(spacing+"Predict",prediction)
        return

    # Print the question at this node
    question = node.question
    if node.type == 'N':  # IF not categorical split, will ONLY contain 2 branches (True branch/False branch)
        print(spacing + question.get())

        # Call this function recursively to print true branch
        print(spacing + '--> True:')
        printTree(node.branches[0], spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        printTree(node.branches[1], spacing + "  ")

    else:   # IF categorical split, will potentially contain MORE THAN 2 split
        questions = question.get()
        for i in range(len(questions)):
            print(spacing + questions[i])

            # Call this function recursively to print true branch
            print(spacing + '--> True:')
            printTree(node.branches[i], spacing + "  ")


def printOutput(train_ds=None, model=None, test_ds=None, predictions=None, accuracy=None, oob=None, oob_predictions=None, oob_accuracy=None):
    if train_ds is not None:
        print('\nTraining Dataset:\n{}'.format(train_ds))

    if model is not None:
        if isinstance(model,list):
            count = 0
            print('\nRandom Forest', end='')
            for tree in model:
                count += 1
                print('\nTree {}:'.format(count))
                printTree(tree)
        else:
            print('\nDecision Tree:')
            printTree(model)

    if test_ds is not None:
        print('\nTest Dataset:\n{}'.format(test_ds))

    if predictions is not None:
        print('\nPredictions:')

        # Format to 2 decimal places
        for p in predictions:
            for label in p:
                p[label] = round(p[label],2)
            print(p)

    if accuracy is not None:
        print('\nAccuracy: {:.2f}%'.format(accuracy))

    if oob is not None:
        print('\nOOB Rows:\n{}'.format(oob))

    if oob_predictions is not None:
        print('\nOOB Predictions:')

        # Format to 2 decimal places
        for oob_p in oob_predictions:
            for label in oob_p:
                oob_p[label] = round(oob_p[label],2)
            print(oob_p)

    if oob_accuracy is not None:
        print('\nOut-of-bag(OOB) Accuracy: {:.2f}%'.format(oob_accuracy))