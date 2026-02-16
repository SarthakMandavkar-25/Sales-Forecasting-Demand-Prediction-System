import os 
import sys
import pandas as pd 
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass 

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
# This function will do all the transformation on the dataset.
        try:
            logging.info("Defining numerical and categorical columns in dataset")
            numerical_columns = [
                "year",
                "month",
                "day",
                "day_of_week",
                "day_of_year",
                "week_of_year",
                "quarter",
                "is_weekend",
                "is_holiday",
                "store_id",
                "category_id",
                "base_price",
                "discount_percentage",
                "is_promotion",
                "price",
                "stock_level",
                "competitor_price",
                "temperature",
                "precipitation",
                "gdp_growth",
                "unemployment_rate",
                "consumer_confidence",
                "marketing_spend",
                "website_visits",
                "social_media_engagement",
                "customer_footfall"
            ]

            categorical_columns = [
                "store_name",
                "product_category",
                "product_subcategory"
            ]

# Creating Pipelines (numerical pipeline for numerical colimns & categorical pipeline for categorical columns)
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'most_frequent')),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))

                ]
            )

            logging.info(f"Numerical Columns are : {numerical_columns}")
            logging.info(f"Categorical Columns are : {categorical_columns}")

# Margeing these two Pipelines by ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ],
                remainder = 'drop'
            )

            logging.info("Column Transformation Done")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Entered into Data Transformation Compo")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train test data completed")
            logging.info("Obtaining Preprocessor object")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = "sales"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor abject on training and testing dataset")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                ]

            logging.info("Saving Preprocessing Object")

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)