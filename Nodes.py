class Node:  # Contains test-question, and branches
    def __init__(self,question,branches,type):
        self.question = question
        self.branches = branches
        self.type = type

class Leaf:  # Holds a dictionary to classify sample
    def __init__(self, ds=None, labels=None, prediction=None):
        if ds is not None and labels is not None:
            self.prediction = {}
            for label in labels:
                self.prediction[label] = (len(ds[ds.Label == label].index) / len(ds.index))
        elif prediction is not None:
            self.prediction = prediction