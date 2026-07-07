# City of Toronto Bike Theft Recovery Prediction

## Project Description

This project uses Toronto Police bicycle theft open data to predict whether a stolen bicycle is likely to be recovered. It includes data cleaning, exploratory analysis, Power BI dashboarding, machine learning model comparison, and Flask API deployment.

## Project Objectives

- Analyze bicycle theft patterns in Toronto
- Build a model to classify theft cases as `STOLEN` or `RECOVERED`
- Handle severe class imbalance in the dataset
- Visualize key trends using Python, Power BI, and Tableau
- Deploy the final model through a Flask API

## Dataset

The dataset contains bicycle theft records from 2013 to 2025, including theft dates, locations, bicycle details, offence information, and theft status.

Key details:

- Raw records: 39,473
- Target variable: `STATUS`
- Target classes: `STOLEN` and `RECOVERED`
- Recovery rate: approximately 1%

## Tools and Technologies

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn
- Power BI
- Tableau
- Flask
- Pickle
- Postman

## Data Exploration

Exploratory analysis was used to understand theft trends and data quality issues.

Key findings:

- Theft counts increased from 2016 to 2020
- Most thefts occurred in the afternoon and evening
- Theft activity was highest in summer months
- Many thefts happened on streets, parking areas, and outside commercial buildings
- Recovery rate was extremely low, creating a strong class imbalance

## Data Cleaning and Preprocessing

Main cleaning and preprocessing steps:

- Filtered records to the 2013–2025 period
- Handled missing categorical and numeric values
- Corrected invalid coordinates, negative costs, and unrealistic outliers
- Removed high-cardinality and non-predictive fields
- Encoded categorical variables
- Scaled numeric variables
- Applied `SelectKBest` for feature selection

## Power BI Dashboard
<img width="1037" height="557" alt="Screenshot 2026-07-07 at 2 24 46 PM" src="https://github.com/user-attachments/assets/333d148d-ad54-4baf-8829-0f5ab848dc5e" />


## Machine Learning Approach

Two classification models were trained and compared:

- Logistic Regression
- Decision Tree Classifier

The dataset was highly imbalanced, so SMOTE-ENN was used to resample the training data.

The modeling workflow included:

1. Data preprocessing
2. Feature selection
3. Class imbalance handling
4. Model training
5. Model evaluation
6. Final model selection

## Model Evaluation

Logistic Regression performed slightly better than the Decision Tree model based on ROC AUC.

| Model | ROC AUC |
|---|---:|
| Logistic Regression | 0.631 |
| Decision Tree | 0.617 |

Logistic Regression was selected as the final model for deployment.

## Model Deployment

The final model was deployed using Flask.

The deployment includes:

- Pickle model bundle
- Preprocessing pipeline
- Feature selector
- Logistic Regression model
- Flask API
- HTML form for manual testing

Main endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/predict` | POST | Returns prediction from JSON input |
| `/ui` | GET | Displays HTML input form |
| `/ui/result` | POST | Returns prediction from form input |

## API Output

Example API response:

```json
{
  "prediction_label": "NOT RETURNED (stolen)",
  "prediction_class": 1,
  "probability_not_returned": 0.94
}
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/thuhale2210/comp309-final-project.git
cd comp309-final-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open the browser UI

```text
http://127.0.0.1:5000/ui
```

You can also test the prediction API using Postman:

```text
POST http://127.0.0.1:5000/predict
```

## Key Results

- Built a complete machine learning pipeline for bicycle theft recovery prediction
- Created Python visualizations and an interactive Power BI dashboard for theft pattern analysis
- Handled severe class imbalance using SMOTE-ENN
- Compared Logistic Regression and Decision Tree models using ROC AUC
- Selected Logistic Regression as the final model with an ROC AUC of 0.631
- Deployed the model through a Flask API with Pickle serialization
- Tested the API using Postman and a browser-based UI
