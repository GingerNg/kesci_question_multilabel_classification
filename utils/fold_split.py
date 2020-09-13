from sklearn.model_selection import train_test_split
from sklearn import utils
import random

class Slicer(object):
    
    def split(self):
        pass

class SklearnSlicer(Slicer):
    def split(self, data, fea_col_name, label_col_name):
        #划分训练集和测试集
        X = data[fea_col_name]
        y = data[label_col_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        return X_train, X_test, y_train, y_test


class ShuffleSlicer(Slicer):
    def split(self, df):
        if isinstance(df, list):
            df = random.shuffle(df)
        else:
            df = utils.shuffle(df)
        train_len = df.shape[0]//5 * 4
        train_df = df[0:train_len]
        test_df = df[train_len:]
        return train_df, test_df