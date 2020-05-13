import numpy as np

class Question:  # Record test-split used to split dataset at each node
    def __init__(self,feature,feature_value):
        self.f = feature
        self.f_val = feature_value


    def split(self,ds):  # Split dataset using test-split
        if isinstance(self.f_val,np.int64) or isinstance(self.f_val,np.float64):
            ds1 = ds[ds[self.f] <= self.f_val].copy()
            ds2 = ds[ds[self.f] > self.f_val].copy()
        else:
            ds1 = ds[ds[self.f] == self.f_val].copy()
            ds2 = ds[ds[self.f] != self.f_val].copy()

        ds1.reset_index(drop=True, inplace=True)
        ds2.reset_index(drop=True, inplace=True)
        return ds1,ds2


    def match(self,row,value=None):  # Match test split & decide to proceed with true or false branch
        if value is None:
            value = self.f_val

        val = row[self.f]
        if isinstance(val,np.int64) or isinstance(val,np.float64):
            return val <= value
        else:
            return val == value


    # For VISUALIZATION purposes only
    def get(self):
        if isinstance(self.f_val,np.int64) or isinstance(self.f_val,np.float64):
            return 'Is {0} <= {1}?'.format(self.f, self.f_val)

        else:   # If CATEGORICAL, f_val might be a string or a list when this function is called
            if isinstance(self.f_val,str):
                return 'Is {0} == {1}?'.format(self.f,self.f_val)
            elif isinstance(self.f_val,list):
                questions = []
                for val in self.f_val:
                    questions.append('Is {0} == {1}?'.format(self.f,val))
                return questions
