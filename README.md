# COVID-19 Outbreak Prediction

This project predicts COVID-19 new cases using machine learning and deep learning models, supporting SDG 3 (Good Health and Well-being). The workflow includes data preprocessing, feature engineering, model training (Random Forest and LSTM), evaluation, and saving the best model.

## Project Structure
- `covid_prediction.ipynb`: Main Jupyter notebook with all code and analysis.
- `owid-covid-data.csv`: COVID-19 dataset from Our World in Data.
- `covid_new_cases_model.pkl`: Trained Random Forest model for predicting new cases.

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
- joblib

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras joblib
```

## Workflow Overview
1. **Data Loading**: Loads COVID-19 data from `owid-covid-data.csv`.
2. **Preprocessing**: Selects relevant countries and features, handles missing values, and creates time-based features.
3. **Feature Engineering**: Adds lagged features and computes vaccination rates.
4. **Model Training**:
   - **Random Forest Regressor**: Predicts new cases, with hyperparameter tuning via GridSearchCV.
   - **LSTM (Deep Learning)**: Time-series forecasting of new cases.
5. **Evaluation**: Reports metrics (MSE, R², RMSE) and visualizes predictions.
6. **Model Saving**: Best Random Forest model is saved as `covid_new_cases_model.pkl`.

## Usage
1. Place `owid-covid-data.csv` in the same directory as the notebook.
2. Open and run `covid_prediction.ipynb` in Jupyter Notebook or JupyterLab.
3. The notebook will guide you through data exploration, model training, and evaluation.

## Results
- Achieves high R² (>0.95) for new case prediction.
- Feature importance and visualizations included in the notebook.

