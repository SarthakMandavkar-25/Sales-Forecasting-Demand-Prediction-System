from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create Flask Application
application = Flask(__name__)

app = application

## Route for Home Page
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            year=int(request.form.get('year')),
            month=int(request.form.get('month')),
            day=int(request.form.get('day')),
            day_of_week=int(request.form.get('day_of_week')),
            day_of_year=int(request.form.get('day_of_year')),
            week_of_year=int(request.form.get('week_of_year')),
            quarter=int(request.form.get('quarter')),
            is_weekend=int(request.form.get('is_weekend')),
            is_holiday=int(request.form.get('is_holiday')),
            store_id=int(request.form.get('store_id')),
            store_name=request.form.get('store_name'),
            category_id=int(request.form.get('category_id')),
            product_category=request.form.get('product_category'),
            product_subcategory=request.form.get('product_subcategory'),
            base_price=float(request.form.get('base_price')),
            discount_percentage=float(request.form.get('discount_percentage')),
            is_promotion=int(request.form.get('is_promotion')),
            price=float(request.form.get('price')),
            stock_level=int(request.form.get('stock_level')),
            competitor_price=float(request.form.get('competitor_price')),
            temperature=float(request.form.get('temperature')),
            precipitation=float(request.form.get('precipitation')),
            gdp_growth=float(request.form.get('gdp_growth')),
            unemployment_rate=float(request.form.get('unemployment_rate')),
            consumer_confidence=float(request.form.get('consumer_confidence')),
            marketing_spend=float(request.form.get('marketing_spend')),
            website_visits=int(request.form.get('website_visits')),
            social_media_engagement=int(request.form.get('social_media_engagement')),
            customer_footfall=int(request.form.get('customer_footfall'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=round(results[0], 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0")