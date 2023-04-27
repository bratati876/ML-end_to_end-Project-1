# this file is used to write the common functionalities

import os
import sys

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics  import r2_score, mean_absolute_error, mean_squared_error

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model( X_train, y_train, X_test, y_test, models):
    # mae = mean_absolute_error(true, predicted)
    # mse = mean_squared_error(true, predicted)
    # rmse = np.sqrt(mean_squared_error(true, predicted))
    # r2_square = r2_score(true, predicted)
    # return mae, rmse, r2_square

    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Train_model
            model.fit(X_train, y_train)
            # Predict for training data
            y_train_pred = model.predict(X_train)
            # Predict for testing data
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report

        
    except Exception as e:
        raise CustomException(e, sys)
        logging.info("Exception occured in the model evaluation code")

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in the load object")
        raise CustomException(e, sys)