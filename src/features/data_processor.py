import numpy as np
import pandas as pd

"""DataProcessor class definition to do the following

1. Remove certain columns
2. Add some columns related to date time
3. Deal with missing values
4. One hot encoding


 """


class DataProcessor:
    def __init__(self, cols_to_remove=None, datecol=None):
        self.cols_to_remove = cols_to_remove
        self.datecol = datecol
        self.was_fit = False

    def fit(self, X, y=None):
        """fit the process on the training data"""

        self.was_fit = True

        # remove the columns
        X_new = X.drop(columns=self.cols_to_remove, axis=1)

        # get the categorical features
        self.categorical_features = X_new.dtypes[X_new.dtypes == "object"].index

        # dummy encoding
        dummy_df = pd.get_dummies(
            X_new, columns=self.categorical_features, dummy_na=True
        )
        self.allcols = dummy_df.columns

        return self

    def transform(self, X, y=None):
        """transform the process on the train/test data """

        if not self.was_fit:
            raise Error("Fit the DataProcessor first")

        # remove the columns
        X_new = X.drop(columns=self.cols_to_remove, axis=1)

        # get the categorical features
        self.categorical_features = X_new.dtypes[X_new.dtypes == "object"].index

        # dummy encoding
        X_new = pd.get_dummies(X_new, columns=self.categorical_features, dummy_na=True)

        # this is for test - make sure the dummy columns that are not in test 
        # but present in train are set to 0
        newcols = set(self.allcols) - set(X_new.columns)
        if newcols:
            for col in newcols:
                X_new[col] = 0

        X_new = X_new[self.allcols]

        # Create month and year columns for the transactiondate 
        # and drop transactiondate
        if self.datecol:
            X_new[self.datecol + "_month"] = pd.to_datetime(
                X_new[self.datecol]
            ).dt.month
            X_new[self.datecol + "_year"] = pd.to_datetime(X_new[self.datecol]).dt.year
            X_new = X_new.drop(columns=self.datecol, axis=1)

        # fill NaN with -1
        X_new = X_new.fillna(-1)

        return X_new

    def fit_transform(self, X, y=None):
        """fit and transform"""

        return self.fit(X).transform(X)
