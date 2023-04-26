## import the libraries 
import os
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from src.utils import evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting dependent and independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'linear_regression':LinearRegression(),
                'ridge_regression':Ridge(),
                'lasso_regression':Lasso(),
                'Elastic net':ElasticNet()
            }

            model_report:dict  = evaluate_model( X_train, y_train, X_test, y_test , models)

            print(model_report)

            print("--------------------------------------------------")
            logging.info(f'Model Report:{model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f'Best model is found : {best_model_name} , with the r2 score :{best_model_score}')


            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            pass
        except Exception as e:
            logging.info("Exception occured at initiate_model_training")
            raise CustomException(e, sys)