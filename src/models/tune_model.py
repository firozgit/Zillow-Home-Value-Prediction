import sys
import inspect

#Add the scripts directory to the sys path
sys.path.append("../features")
from data_processor import DataProcessor

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

"""
modelfit function - Gets the number of boosting rounds using XGBoost Cross Validation Method
xgb_gridsearch function - Get the best parameters of the XGBoost model using Pipeline, GridSearchCV and DataProcessor class

"""



def modelfit(xgb_reg, X_train, y_train, X_test, y_test):
    """
    Get the number of boosting rounds of the model using XGBoost Cross Validation Method
    
    Use the above parameters to Fit the XGBoost model on the train and get feature importance and predictons
       
    Parameters:
    xgb_reg model, X_train, y_train, X_test, y_test datasets
    
    
    Returns:
    None
        
    """
    
    xgb_param = xgb_reg.get_xgb_params()
    
    num_boost_round = xgb_reg.get_params()['n_estimators']
    num_boost_round

    xgb_train = xgb.DMatrix(X_train, y_train) 

    cv_results = xgb.cv(xgb_param, xgb_train, num_boost_round=num_boost_round, nfold=3, metrics='mae', early_stopping_rounds=50)

    # print("\n---- cv results ----\n\n", cv_results)
    # print("\nCV test results best - MAE: {}, ".format(cv_results['test-mae-mean'].min()))
    print("\nBest estimators :", cv_results.shape[0])
    
    xgb_reg.set_params(n_estimators=cv_results.shape[0])

    xgb_reg.fit(X_train, y_train, eval_metric='mae')

    dtrain_pred = xgb_reg.predict(X_train)

    print("\nTrain MAE: {}".format(mean_absolute_error(y_train, dtrain_pred)))
    
    dtest_pred = xgb_reg.predict(X_test)
    
    print("\nTest MAE: {}".format(mean_absolute_error(y_test, dtest_pred)))

    plt.figure(figsize=(20,15))
    xgb.plot_importance(xgb_reg, ax=plt.gca())
    
    return None
    

def xgb_gridsearch(param_test, xgb_reg):
    """
    Get the best parameters of the XGBoost model using Pipeline, GridSearchCV and DataProcessor class
    
    Parameters:
    param_test, xgb_reg model
    
    
    Returns:
    None
        
    """
    
    dp = DataProcessor(cols_to_remove=["parcelid", "propertyzoningdesc", "rawcensustractandblock", "regionidneighborhood", "regionidzip", "censustractandblock"], 
                      datecol="transactiondate")

    pipeline = Pipeline([
        ("dataprocessor", dp),
        ("xgb_reg", xgb_reg)
    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_test, scoring="neg_mean_absolute_error", 
                               n_jobs=-1, cv=3, verbose=1)

    grid_search.fit(X_train, y_train)

    print("----- Grid Search cv results ----- \n")
    for mean_score, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
        print(-(mean_score), params)

    print("\n----- Grid Search best parameters ------ \n", grid_search.best_params_)
    print("\n")
    print("----- Grid Search best score ------ \n", -(grid_search.best_score_))
    
    return None