import os
import sys
import pandas as pd 
import numpy as np 

from dataclasses import dataclass
from src.utils import load_object, evaluate_models, save_object
from src.logger import logging 
from src.exception import CustomException 

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

# from catboost import CatBoostRegressor

from sklearn.metrics import r2_score, accuracy_score

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Entered into Model Trainer compo.")
        try:
            logging.info("Split Training and Testing inpuit data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

# Initializing Algorithms 
            models = {
                "Random Forest"      : RandomForestRegressor(),
                "Gradient Boosting"  : GradientBoostingRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "XGBRegressor"       : XGBRegressor(),
                # "CatBoost Regressor" : ,
                # "LightGBM"
            }
            
#  HyperParameters 
            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                # "CatBoost Regressor": {          
                #     'iterations': [100, 200],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'depth': [4, 6, 8]
                # },
                # "LightGBM": {                    
                #     'n_estimators': [100, 200],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'num_leaves': [20, 31, 50]
                # }
            }

            model_report : dict = evaluate_models(X_train=X_train, X_test=X_test,
                                                  y_train=y_train, y_test=y_test,
                                                  param=params, models=models)
            
# To get best model score
            best_model_score = max(sorted(model_report.values()))

# To get Best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best found model on both training and testing datasets")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return best_model, X_test, y_test, r2_square

        except Exception as e:
            raise CustomException(e, sys)