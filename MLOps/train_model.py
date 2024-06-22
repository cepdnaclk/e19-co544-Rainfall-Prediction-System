'''# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
df_original = pd.read_excel('transformed_rainfall_data.xlsx')
df = df_original[~df_original.index.isin([0, 1])]

# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# Extracting day of the year as a feature
df['DayOfYear'] = df['Date'].dt.dayofyear

# Define targets and features
targets = ['Vavuniya', 'Anuradhapura', 'Maha Illuppallama']
features = [
    'DayOfYear',
    'Vavuniya_lag1', 'Anuradhapura_lag1', 'Maha Illuppallama_lag1',
    'Vavuniya_rolling_mean_7', 'Vavuniya_rolling_std_7', 'Vavuniya_rolling_median_7',
    'Anuradhapura_rolling_mean_7', 'Anuradhapura_rolling_std_7', 'Anuradhapura_rolling_median_7',
    'Maha Illuppallama_rolling_mean_7', 'Maha Illuppallama_rolling_std_7', 'Maha Illuppallama_rolling_median_7'
]

# Lag features (previous day's value)
for target in targets:
    df[f'{target}_lag1'] = df[target].shift(1)

# Calculate rolling statistics
window_size = 7

for target in targets:
    df[f'{target}_rolling_mean_{window_size}'] = df[target].rolling(window=window_size).mean()
    df[f'{target}_rolling_std_{window_size}'] = df[target].rolling(window=window_size).std()
    df[f'{target}_rolling_median_{window_size}'] = df[target].rolling(window=window_size).median()

# After feature engineering
df.dropna(inplace=True)

# Select the features from the DataFrame
X = df[features]
y_vavuniya = df['Vavuniya']
y_anuradhapura = df['Anuradhapura']
y_maha = df['Maha Illuppallama']

X_train, X_test, y_train_v, y_test_v = train_test_split(X, y_vavuniya, test_size=0.2, random_state=42)
X_train, X_test, y_train_a, y_test_a = train_test_split(X, y_anuradhapura, test_size=0.2, random_state=42)
X_train, X_test, y_train_m, y_test_m = train_test_split(X, y_maha, test_size=0.2, random_state=42)

# Train and evaluate the models
def train_and_evaluate_model(X_train, X_test, y_train, y_test, target_name):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model performance for {target_name}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print()

    return model, mse, mae, r2

model_vavuniya, mse_v, mae_v, r2_v = train_and_evaluate_model(X_train, X_test, y_train_v, y_test_v, 'Vavuniya')
model_anuradhapura, mse_a, mae_a, r2_a = train_and_evaluate_model(X_train, X_test, y_train_a, y_test_a, 'Anuradhapura')
model_maha, mse_m, mae_m, r2_m = train_and_evaluate_model(X_train, X_test, y_train_m, y_test_m, 'Maha Illuppallama')

# Write model scores to file
with open("scores.txt", 'w') as score_file:
    score_file.write("Vavuniya Model Performance:\n")
    score_file.write(f"Mean Absolute Error: {mae_v}\n")
    score_file.write(f"Mean Squared Error: {mse_v}\n")
    score_file.write(f"R-squared: {r2_v}\n\n")

    score_file.write("Anuradhapura Model Performance:\n")
    score_file.write(f"Mean Absolute Error: {mae_a}\n")
    score_file.write(f"Mean Squared Error: {mse_a}\n")
    score_file.write(f"R-squared: {r2_a}\n\n")

    score_file.write("Maha Illuppallama Model Performance:\n")
    score_file.write(f"Mean Absolute Error: {mae_m}\n")
    score_file.write(f"Mean Squared Error: {mse_m}\n")
    score_file.write(f"R-squared: {r2_m}\n\n")
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
df_original = pd.read_csv('transformed_rainfall_data.csv')
df = df_original[~df_original.index.isin([0, 1])]

# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# Extracting day of the year as a feature
df['DayOfYear'] = df['Date'].dt.dayofyear

# Define targets and features
targets = ['Vavuniya', 'Anuradhapura', 'Maha Illuppallama']
features = [
    'DayOfYear',
    'Vavuniya_lag1', 'Anuradhapura_lag1', 'Maha Illuppallama_lag1',
    'Vavuniya_rolling_mean_7', 'Vavuniya_rolling_std_7', 'Vavuniya_rolling_median_7',
    'Anuradhapura_rolling_mean_7', 'Anuradhapura_rolling_std_7', 'Anuradhapura_rolling_median_7',
    'Maha Illuppallama_rolling_mean_7', 'Maha Illuppallama_rolling_std_7', 'Maha Illuppallama_rolling_median_7'
]

# Lag features (previous day's value)
for target in targets:
    df.loc[:, f'{target}_lag1'] = df[target].shift(1)

# Calculate rolling statistics
window_size = 7

for target in targets:
    df.loc[:, f'{target}_rolling_mean_{window_size}'] = df[target].rolling(window=window_size).mean()
    df.loc[:, f'{target}_rolling_std_{window_size}'] = df[target].rolling(window=window_size).std()
    df.loc[:, f'{target}_rolling_median_{window_size}'] = df[target].rolling(window=window_size).median()

# After feature engineering
df.dropna(inplace=True)

# Select the features from the DataFrame
X = df[features]
y_vavuniya = df['Vavuniya']
y_anuradhapura = df['Anuradhapura']
y_maha = df['Maha Illuppallama']

X_train, X_test, y_train_v, y_test_v = train_test_split(X, y_vavuniya, test_size=0.2, random_state=42)
X_train, X_test, y_train_a, y_test_a = train_test_split(X, y_anuradhapura, test_size=0.2, random_state=42)
X_train, X_test, y_train_m, y_test_m = train_test_split(X, y_maha, test_size=0.2, random_state=42)

# Train and evaluate the models
def train_and_evaluate_model(X_train, X_test, y_train, y_test, target_name):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model performance for {target_name}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print()

    return model, mse, mae, r2

model_vavuniya, mse_v, mae_v, r2_v = train_and_evaluate_model(X_train, X_test, y_train_v, y_test_v, 'Vavuniya')
model_anuradhapura, mse_a, mae_a, r2_a = train_and_evaluate_model(X_train, X_test, y_train_a, y_test_a, 'Anuradhapura')
model_maha, mse_m, mae_m, r2_m = train_and_evaluate_model(X_train, X_test, y_train_m, y_test_m, 'Maha Illuppallama')

# Write model scores to file
with open("scores.txt", 'w') as score_file:
    score_file.write("Vavuniya Model Performance:\n")
    score_file.write(f"Mean Absolute Error: {mae_v}\n")
    score_file.write(f"Mean Squared Error: {mse_v}\n")
    score_file.write(f"R-squared: {r2_v}\n\n")

    score_file.write("Anuradhapura Model Performance:\n")
    score_file.write(f"Mean Absolute Error: {mae_a}\n")
    score_file.write(f"Mean Squared Error: {mse_a}\n")
    score_file.write(f"R-squared: {r2_a}\n\n")

    score_file.write("Maha Illuppallama Model Performance:\n")
    score_file.write(f"Mean Absolute Error: {mae_m}\n")
    score_file.write(f"Mean Squared Error: {mse_m}\n")
    score_file.write(f"R-squared: {r2_m}\n\n")

