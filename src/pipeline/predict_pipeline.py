import sys
import os

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            # After transforming, model will do the prediction
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        year: int,
        month: int,
        day: int,
        day_of_week: int,
        day_of_year: int,
        week_of_year: int,
        quarter: int,
        is_weekend: int,
        is_holiday: int,
        store_id: int,
        store_name: str,
        category_id: int,
        product_category: str,
        product_subcategory: str,
        base_price: float,
        discount_percentage: float,
        is_promotion: int,
        price: float,
        stock_level: int,
        competitor_price: float,
        temperature: float,
        precipitation: float,
        gdp_growth: float,
        unemployment_rate: float,
        consumer_confidence: float,
        marketing_spend: float,
        website_visits: int,
        social_media_engagement: int,
        customer_footfall: int
    ):
        self.year = year
        self.month = month
        self.day = day
        self.day_of_week = day_of_week
        self.day_of_year = day_of_year
        self.week_of_year = week_of_year
        self.quarter = quarter
        self.is_weekend = is_weekend
        self.is_holiday = is_holiday
        self.store_id = store_id
        self.store_name = store_name
        self.category_id = category_id
        self.product_category = product_category
        self.product_subcategory = product_subcategory
        self.base_price = base_price
        self.discount_percentage = discount_percentage
        self.is_promotion = is_promotion
        self.price = price
        self.stock_level = stock_level
        self.competitor_price = competitor_price
        self.temperature = temperature
        self.precipitation = precipitation
        self.gdp_growth = gdp_growth
        self.unemployment_rate = unemployment_rate
        self.consumer_confidence = consumer_confidence
        self.marketing_spend = marketing_spend
        self.website_visits = website_visits
        self.social_media_engagement = social_media_engagement
        self.customer_footfall = customer_footfall

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "year":                   [self.year],
                "month":                  [self.month],
                "day":                    [self.day],
                "day_of_week":            [self.day_of_week],
                "day_of_year":            [self.day_of_year],
                "week_of_year":           [self.week_of_year],
                "quarter":                [self.quarter],
                "is_weekend":             [self.is_weekend],
                "is_holiday":             [self.is_holiday],
                "store_id":               [self.store_id],
                "store_name":             [self.store_name],
                "category_id":            [self.category_id],
                "product_category":       [self.product_category],
                "product_subcategory":    [self.product_subcategory],
                "base_price":             [self.base_price],
                "discount_percentage":    [self.discount_percentage],
                "is_promotion":           [self.is_promotion],
                "price":                  [self.price],
                "stock_level":            [self.stock_level],
                "competitor_price":       [self.competitor_price],
                "temperature":            [self.temperature],
                "precipitation":          [self.precipitation],
                "gdp_growth":             [self.gdp_growth],
                "unemployment_rate":      [self.unemployment_rate],
                "consumer_confidence":    [self.consumer_confidence],
                "marketing_spend":        [self.marketing_spend],
                "website_visits":         [self.website_visits],
                "social_media_engagement":[self.social_media_engagement],
                "customer_footfall":      [self.customer_footfall],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)