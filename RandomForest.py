# Python modules
import pandas as pd

# Self-defined modules
import UtilityFunctions as uf
import DecisionTree as dt


# ============================================================ Important Global Variables =========================================================================#
# Dataset Details
ds = pd.read_excel(r'Toy_Dataset.xlsx')  # Label MUST be at LAST column
feature_list = list(ds.columns)[0:-1]
label_list = list(set(ds['Label'][row] for row in range(len(ds.index))))
categorical_vals = {}  # A dictionary contains all categorical feature values (refer for-loop below)

for feature in feature_list:     # e.g. categorical_vals['Color'] = ['Red', 'Blue', 'Green']
    if isinstance(ds[feature][0],str):
        categorical_vals[feature] = list(set([ds[feature][row] for row in range(len(ds[feature].index))]))


# Update decision tree's global variables accordingly
dt.feature_list = feature_list
dt.label_list = label_list
dt.categorical_vals = categorical_vals


# ===================================================================== Methods ====================================================================================#

# Build random forest model (call buildTree() n times for n trees)
def buildForest(bootstrap_ds, n_features=None, min_sample_split=None):
    forest = []
    for i in bootstrap_ds:
        # Initialize important global variables
        dt.chosen_vals = {f: [] for f in dt.feature_list}
        dt.chosen_f = []
        tree = dt.buildTree(bootstrap_ds[i], n_features, min_sample_split)
        forest.append(tree)
    return forest


def classifyOOB(ds_train, forest, oob_dict):  # Test every tree with their respective out-of-bag dataset
    oob_predictions = []

    for i in oob_dict:    # For every out-of-bag row
        row = ds_train.iloc[i]
        oob_tree_list = oob_dict[i]
        currentRow_prediction = dict.fromkeys(label_list, 0)

        for tree in oob_tree_list:      # For every tree corresponding to current out-of-bag row
            tree_prediction = dt.performClassification(forest[tree], row)
            treePrediction_Label = uf.get_PredictionLabel(tree_prediction)

            # Increment count of tree vote
            currentRow_prediction[treePrediction_Label] += 1

        for label in currentRow_prediction:
            currentRow_prediction[label] /= len(oob_tree_list)

        oob_predictions.append(currentRow_prediction)

    return oob_predictions


# Get forest predictions for the supplied test-set
def classify(forest,test_ds):
    predictions = []  # Forest predictions

    for i in range(len(test_ds)):
        currentRow_prediction = dict.fromkeys(label_list,0)

        # For every tree, perform prediction on current test row & sum probability
        for tree in forest:
            tree_prediction = dt.performClassification(tree, test_ds.iloc[i])
            treePrediction_Label = uf.get_PredictionLabel(tree_prediction)

            # Increment count of tree vote
            currentRow_prediction[treePrediction_Label] += 1

        # Divide vote count by no. of trees to get aggregated-probability
        for label in currentRow_prediction:
            currentRow_prediction[label] /= len(forest)

        predictions.append(currentRow_prediction)

    return predictions



def main():
    # Initializa RF Hyperparameters
    n_trees = 3  # Avoid tie-votes
    n_features = None
    min_sample_split = -1

    # Split train/test & create bootstrapped dataset
    train_ds, test_ds = uf.splitTrainTest(ds)
    bootstrap_ds, oob_dict, oob_ds = uf.createBootstrap(train_ds, n_trees)

    # Build RF model and evaluate predictions on test-set/OOB
    forest = buildForest(bootstrap_ds,n_features,min_sample_split)
    predictions = classify(forest,test_ds)
    oob_predictions = classifyOOB(train_ds,forest,oob_dict)
    accuracy = uf.evaluate(predictions,test_ds)
    oob_accuracy = uf.evaluate(oob_predictions,oob_ds)
    uf.printOutput(train_ds=train_ds, model=forest, test_ds=test_ds, predictions=predictions, accuracy=accuracy, oob=oob_ds, oob_predictions=oob_predictions\
                   ,oob_accuracy=oob_accuracy)

    # Save model as 'myforest.txt' in current directory
    uf.saveModel(forest,'myforest.txt')

    # Load model from 'myforest.txt'
    loaded_forest = uf.loadModel('myforest.txt')


if __name__ == '__main__':
    main()