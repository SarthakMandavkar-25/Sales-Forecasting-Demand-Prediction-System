## Sales Forecasting & Demand Prediction System.
A machine learning-powered system that forecasts future sales and predicts product demand using historical data, enabling smarter inventory management, resource planning, and business decision-making.

#### Problem Statement
Businesses often struggle to accurately estimate future demand for products. Incorrect demand forecasts can lead to:
- Overstocking
- Stock shortages
- Increased inventory costs
- Revenue loss

The objective of this project is to develop a predictive system that forecasts future sales and demand using historical sales data.

#### Objectives
- Predict future sales quantities.
- Forecast product demand.
- Assist in inventory planning.
- Improve business decision-making.
- Reduce operational costs.

#### Technologies Used 
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Flask
- HTML
- CSS

#### Machine Learning Workflow
##### Data Collection
Historical sales data is collected and stored in CSV format.

##### Data Preprocessing
- Handling missing values
- Removing duplicates
- Feature engineering
- Data cleaning

##### Exploratory Data Analysis (EDA)
- Sales trend analysis
- Product demand analysis
- Seasonal pattern analysis
- Correlation analysis

##### Feature Engineering
Input Features = 29
Data Categories = 7 

Numerical Features = year, month, day, day_of_week, day_of_year, week_of_year,quarter, is_weekend, is_holiday, store_id, category_id, product_subcategory, base_price, discount_percentage, price,stock_level, competitor_price, temperature,precipitation, gdp_growth, unemployment_rate, consumer_confidence, marketing_spend,website_visits, social_media_engagement, customer_footfall, Sales

Categorical Features = store_name, product_category, product_subcategory, is_promotion

##### Model Training
The model is trained using historical sales data.
Algorithms:
- XGBoost Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- LightBoost Regressor
- CatBoost Regressor

##### Model Evaluation
Metrics:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

#### Project Architecture
User Input
      ↓
Flask Web Application
      ↓
Data Preprocessing
      ↓
Trained ML Model
      ↓
Sales Prediction
      ↓
Demand Forecast Output

#### Applications 
- Retail Stores
- E-Commerce Platforms
- Inventory Management
- Supply Chain Optimization
- Manufacturing Industries

#### Project Structure 
sales-forecasting/
├── artifacts/
│   ├── data.csv/                  # Raw sales data
│   ├── train.csv/                 # Train data (80%)
│   ├── test.csv/                  # Test data  (20%)
|   ├── model.pkl/                 # Trained model pickle file 
|   ├── preprocessor.pkl/          # Cleaned and feature-engineered pickle file
|   └── metrics.pkl/               # Model evaluation pickle file 
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb     
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb        # MAE, R2-Score, MSE, etc.
│     
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      
│   │   ├── data_transformation.py 
|   |   ├── model_trainer.py
|   |   └── model_evaluation.py
|   |
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │   
│   ├── logger.py
│   ├── exception.py       
│   └── utils.py       
│   
├── templates/                     # Front-end setup
|   ├── index.html
|   └── home.html
|
├──application.py                  # Flask application    
├── Dockerfile
├── setup.py
├── requirements.txt
└── README.md

#### Evaluation Metrics
The system evaluates forecasts using:

- MAE — Mean Absolute Error.
- RMSE — Root Mean Squared Error.
- MAPE — Mean Absolute Percentage Error.
- R² — Coefficient of Determination.