from flask import Flask, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('covid_new_cases_model.pkl')

@app.route('/')
def dashboard():
    # Load and preprocess data as in your notebook
    df = pd.read_csv("owid-covid-data.csv")
    features = ['date','location','total_cases', 'new_cases', 'total_vaccinations', 'population', 'people_vaccinated']
    countries = ['Africa', 'United_States', 'India', 'United_Kingdom', 'Germany']
    df = df[df['location'].isin(countries)][features]
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['total_vaccinations'] = df['total_vaccinations'].fillna(0)
    df['people_vaccinated'] = df['people_vaccinated'].fillna(0)
    df['total_cases'] = df['total_cases'].fillna(0)
    df['new_cases'] = df['new_cases'].fillna(df['new_cases'].mean())
    for lag in range(1, 8):
        df[f'total_cases_lag_{lag}'] = df.groupby('location')['total_cases'].shift(lag)
        df[f'new_cases_lag_{lag}'] = df.groupby('location')['new_cases'].shift(lag)
    df['vaccination_rate'] = df['people_vaccinated'] / df['population'] * 100
    df = df.dropna()

    # Prepare features for prediction (same as in notebook)
    X = df[['total_cases', 'total_vaccinations', 'vaccination_rate', 'year', 'month', 'day', 'day_of_year'] +
           [f'total_cases_lag_{lag}' for lag in range(1, 8)] +
           [f'new_cases_lag_{lag}' for lag in range(1, 8)]]

    # Predict new cases
    df['predicted_new_cases'] = model.predict(X)

    # Show last 100 rows with predictions
    data = df.head(100)
    return render_template('dashboard.html', data=data.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)