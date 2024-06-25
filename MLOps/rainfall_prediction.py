import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Read the .xlsx file using Pandas
df_original = pd.read_excel('transformed_rainfall_data.xlsx')
print(df_original.head())

# Assuming your dataframe is named 'df'
df = df_original[~df_original.index.isin([0, 1])]

'''# Select features
features = ['Year', 'Month', 'Day']

print(features)'''

#Check for null values
df.isna().sum()

df.head()

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
# Create additional features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfYear'] = df['Date'].dt.dayofyear

# Create lag features and rolling mean features for each station
stations = ['Vavuniya', 'Anuradhapura', 'Maha Illuppallama']
for station in stations:
    df[f'{station}_lag1'] = df[station].shift(1)
    df[f'{station}_lag2'] = df[station].shift(2)
    df[f'{station}_lag3'] = df[station].shift(3)
    df[f'{station}_rolling_mean3'] = df[station].rolling(window=3).mean()
    df[f'{station}_rolling_mean7'] = df[station].rolling(window=7).mean()
df.head()

# Drop the rows with NaN values created by the shift operation
df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

# Features specific to each target
features_vavuniya = [
    'DayOfYear',
    'Year',
    'Vavuniya_lag1',
    'Vavuniya_lag2',
    'Vavuniya_lag3',
    'Vavuniya_rolling_mean3',
    'Vavuniya_rolling_mean7'
]

features_anuradhapura = [
    'DayOfYear',
    'Year',
    'Anuradhapura_lag1',
    'Anuradhapura_lag2',
    'Anuradhapura_lag3',
    'Anuradhapura_rolling_mean3',
    'Anuradhapura_rolling_mean7'
]

features_maha = [
    'DayOfYear',
    'Year',
    'Maha Illuppallama_lag1',
    'Maha Illuppallama_lag2',
    'Maha Illuppallama_lag3',
    'Maha Illuppallama_rolling_mean3',
    'Maha Illuppallama_rolling_mean7'
    
]

# Select the features from the DataFrame
X_vavuniya = df[features_vavuniya]
X_anuradhapura = df[features_anuradhapura]
X_maha = df[features_maha]

y_vavuniya = df['Vavuniya']
y_anuradhapura = df['Anuradhapura']
y_maha = df['Maha Illuppallama']

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_vavuniya, y_vavuniya, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_anuradhapura, y_anuradhapura, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_maha, y_maha, test_size=0.2, random_state=42)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, target_name):
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model , param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    '''best_params = grid_search.best_params_
    model = GradientBoostingRegressor(**best_params)'''
    best_gbr = grid_search.best_estimator_
    y_pred = best_gbr.predict(X_test)

    '''model.fit(X_train, y_train)
    y_pred = model.predict(X_test)'''

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model performance for {target_name}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print()

    return best_gbr
# Train and evaluate models for each target
model_vavuniya = train_and_evaluate_model(X_train_v, X_test_v, y_train_v, y_test_v, 'Vavuniya')
model_anuradhapura = train_and_evaluate_model(X_train_a, X_test_a, y_train_a, y_test_a, 'Anuradhapura')
model_maha = train_and_evaluate_model(X_train_m, X_test_m, y_train_m, y_test_m, 'Maha Illuppallama')

import joblib
joblib.dump(model_vavuniya, 'model_vavuniya.joblib')
joblib.dump(model_anuradhapura, 'model_anuradhapura.joblib')
joblib.dump(model_maha, 'model_maha.joblib')

