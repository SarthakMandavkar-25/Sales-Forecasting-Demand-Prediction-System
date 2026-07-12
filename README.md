# Sales Forecasting & Demand Prediction System

A production-structured machine learning application that forecasts
future sales and predicts product demand from historical sales data —
helping businesses avoid overstocking, stock shortages, and reactive
inventory decisions instead of guessing from gut feel.

---

## Why this is more than a toy notebook

- **Modular pipeline architecture** — ingestion, transformation, model
  training, and evaluation are separate, independently testable
  components (`src/components`, `src/pipeline`), not one long notebook.
- **Config-driven feature set** — 29 engineered input features across
  7 categories (calendar, pricing, inventory, competitor, macroeconomic,
  marketing, and demand-signal features) feeding a reproducible
  preprocessing pipeline.
- **Centralized logging + custom exceptions** — every component logs to
  file and raises a `CustomException` with the exact file and line
  number, the same debugging pattern used in production ML systems.
- **Multiple candidate models, evaluated head-to-head** — XGBoost,
  AdaBoost, Gradient Boosting, LightGBM, and CatBoost regressors are
  trained and compared on MAE, MSE, RMSE, and R².
- **Real web app, not just a script** — Flask application with an
  input form and prediction endpoint, not just a `.predict()` call in
  a notebook.
- **Actually deployable** — Deployed on Render.
  **Live Demo :** - https://sales-forecasting-demand-prediction-53fr.onrender.com

---

## Architecture

```
                    OFFLINE (run once / when data changes)
 data.csv  ──▶  DataIngestion  ──▶  DataTransformation  ──▶  ModelTrainer  ──▶  ModelEvaluation
(raw sales)     (train/test split)  (clean + engineer          (fit XGBoost,        (MAE, MSE,
                                      29 features)               AdaBoost, GBR,       RMSE, R²)
                                                                  LightGBM, CatBoost)
                                                                        │
                                                                        ▼
                                                          model.pkl + preprocessor.pkl


                    ONLINE (every user request)
 User Input ──▶ Flask Web App ──▶ Data Preprocessing ──▶ Trained ML Model ──▶ Sales Prediction
                                  (preprocessor.pkl)      (model.pkl)              │
                                                                                    ▼
                                                                          Demand Forecast Output
```

**Why the split into offline/online pipelines?** Training on the full
historical dataset is slow and only needs to happen when new sales data
comes in. Keeping it out of the request path means every prediction
call is fast — it only runs the saved preprocessor and a single
model inference.

---

## Problem Statement

Businesses often struggle to accurately estimate future demand for
products. Incorrect demand forecasts can lead to overstocking, stock
shortages, increased inventory costs, and lost revenue. This project
builds a predictive system that forecasts future sales and demand from
historical sales data to support smarter inventory management, resource
planning, and business decision-making.

**Objectives**
- Predict future sales quantities
- Forecast product demand
- Assist in inventory planning
- Improve business decision-making
- Reduce operational costs

**Applications**
- Retail stores
- E-commerce platforms
- Inventory management
- Supply chain optimization
- Manufacturing industries

---

## Project Structure

```
sales-forecasting/
├── artifacts/
│   ├── data.csv                   # Raw sales data
│   ├── train.csv                  # Train data (80%)
│   ├── test.csv                   # Test data (20%)
│   ├── model.pkl                  # Trained model
│   ├── preprocessor.pkl           # Fitted preprocessing pipeline
│   └── metrics.pkl                # Model evaluation results
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb         # MAE, RMSE, R², etc.
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Loads + splits raw sales data
│   │   ├── data_transformation.py  # Cleans data, engineers 29 features
│   │   ├── model_trainer.py        # Trains candidate regressors
│   │   └── model_evaluation.py     # Scores models on held-out test set
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py    # OFFLINE: build + evaluate the model
│   │   └── prediction_pipeline.py  # ONLINE: serve a live prediction
│   │
│   ├── logger.py                   # Centralized logging
│   ├── exception.py                # CustomException with file/line tracing
│   └── utils.py                    # Shared helper functions
│
├── templates/                      # Front-end
│   ├── index.html
│   └── home.html
│
├── application.py                  # Flask application entrypoint
├── Dockerfile
├── setup.py
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer               | Choice                                                        | Why |
|----------------------|----------------------------------------------------------------|-----|
| Data handling        | Pandas, NumPy                                                  | Fast, standard tooling for tabular sales data |
| Modeling             | XGBoost, AdaBoost, Gradient Boosting, LightGBM, CatBoost        | Compare boosting algorithms head-to-head, pick the best performer |
| Evaluation           | Scikit-Learn (MAE, MSE, RMSE, R²)                               | Standard regression metrics for forecast accuracy |
| Visualization        | Matplotlib, Seaborn                                             | Trend, seasonality, and correlation analysis in EDA |
| API / Web App        | Flask                                                           | Lightweight, simple to wire a form to a model |
| Frontend             | Plain HTML/CSS                                                  | Zero build step |
| Deployment           | Render                                                          | Portable, reproducible runtime |

---

## Feature Engineering

**Input features = 29 · Data categories = 7**

- **Numerical:** `year`, `month`, `day`, `day_of_week`, `day_of_year`, `week_of_year`, `quarter`, `is_weekend`, `is_holiday`, `store_id`, `category_id`, `product_subcategory`, `base_price`, `discount_percentage`, `price`, `stock_level`, `competitor_price`, `temperature`, `precipitation`, `gdp_growth`, `unemployment_rate`, `consumer_confidence`, `marketing_spend`, `website_visits`, `social_media_engagement`, `customer_footfall`, `Sales`
- **Categorical:** `store_name`, `product_category`, `product_subcategory`, `is_promotion`

---

## Run Locally

```bash
git clone <your-repo-url>
cd sales-forecasting

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Train the model (runs ingestion -> transformation -> training -> evaluation)
python -m src.components.data_ingestion

# Start the Flask app
python app.py
```

Visit **http://127.0.0.1:5000** and enter product/store details to get
a sales prediction and demand forecast.

Run tests: `pytest tests/`

---


## Push to GitHub

```bash
cd sales-forecasting
git init
git add .
git commit -m "Initial commit: Sales Forecasting & Demand Prediction System"
git branch -M main
git remote add origin https://github.com/SarthakMandavkar-25/Sales-Forecasting-Demand-Prediction-System.git
git push -u origin main
```

---

## Extending This Project (nice follow-ups for interviews)

- Add hyperparameter tuning (Optuna/GridSearchCV) and log experiments
  with MLflow.
- Add a `/retrain` endpoint so the model can be refreshed as new sales
  data arrives, instead of retraining manually.
- Swap the static feature set for a streaming pipeline that pulls live
  competitor pricing or weather data.
- Add confidence intervals / prediction ranges, not just point
  forecasts.
- Add a dashboard (Plotly/Streamlit) for visualizing forecast trends
  alongside actuals.

---

## Resume Bullet Points (feel free to adapt)

- Built and deployed a sales forecasting system using XGBoost, LightGBM,
  and CatBoost, with a modular ingestion → transformation → training →
  evaluation pipeline.
- Engineered 29 features across 7 categories (calendar, pricing,
  competitor, macroeconomic, and marketing signals) to improve demand
  forecast accuracy.
- Designed a production-style architecture with centralized logging,
  custom exception handling, and reproducible model artifacts.
- Shipped a Flask web application with a live prediction interface,
  containerized with Docker for deployment.
