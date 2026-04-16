## Sales Forecasting & Demand Prediction System.
A machine learning-powered system that forecasts future sales and predicts product demand using historical data, enabling smarter inventory management, resource planning, and business decision-making.

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