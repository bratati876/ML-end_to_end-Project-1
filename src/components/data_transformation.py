## Importing the libraries
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation is initiated")

            # Define which column should be ordinal encoded and which should be scaled
            categorical_cols=["cut", "color", "clarity"]
            numerical_cols=["carat","depth","table","x","y", "z"]

            # Define the custom ranking for each ordinal variable 
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline initiated")
            ## Numerical Pipeline
            num_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())

            ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
            ('scaler',StandardScaler())
            ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            
            return preprocessor
            logging.info("pipeline completed")
            
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info(f'Train dataframe head: \n {train_df.head().to_string()}')
            logging.info(f'Test dataframe head: \n {test_df.head().to_string()}')

            preprocessor_obj = self.get_data_transformation_object()

            target_column_name = "price"
            drop_columns = [target_column_name, "id"]

            input_feature_train_data = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_data = train_df[target_column_name]

            input_feature_test_data = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_data = test_df[target_column_name]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_data)

            logging.info("Applying the preprocessing upon the train and test data")

            # After we get input features transformed with the preprocessor object we should 
            # convert it from dataframes to numpy arrays for faster computation

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]


            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("pickle file is saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate data tranformation")
            raise CustomException(e, sys)