import os 
import sys 
import pickle
from src.logger import logging 
from src.exception import CustomException 
from dataclasses import dataclass
from src.utils import save_object

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

@dataclass 
class ModelEvaluationConfig:
    model_evaluation_metrics_path : str = os.path.join("artifacts", "metrics.pkl")

class ModelEvaluation:
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self,best_model, X_test, y_test):
        try:
            logging.info("Model Evaluation Starts >>>>")
            y_pred = best_model.predict(X_test)

            # r2_score 
            r2_score_value = r2_score(y_test, y_pred)

            # Mean Square Error 
            mse = mean_squared_error(y_test, y_pred)

            # Mean Absolute Error 
            mae = mean_absolute_error(y_test, y_pred)

            # Root Mean Square Error 
            rmse = mse ** 0.5

            logging.info("Now we create a metrices to show all the results")

            metrics = {
                "r2 Score " : r2_score_value,
                "Mean Squared Error " : mse,
                "Mean Absolute Error " : mae,
                "Root Mean Squared Error " : rmse
            }

            logging.info(f"Evaluation Metrices : {metrics}")

            save_object(
                file_path = self.evaluation_config.model_evaluation_metrics_path,
                obj = metrics
            )

            logging.info("Save the Evaluation metrics Successfully.")

            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)