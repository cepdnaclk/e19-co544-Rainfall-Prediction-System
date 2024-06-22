'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

# Feature Check
expected_features = [
    'DayOfYear', 'Vavuniya_lag1', 'Anuradhapura_lag1', 'Maha Illuppallama_lag1',
    'Vavuniya_rolling_mean_7', 'Vavuniya_rolling_std_7', 'Vavuniya_rolling_median_7',
    'Anuradhapura_rolling_mean_7', 'Anuradhapura_rolling_std_7', 'Anuradhapura_rolling_median_7',
    'Maha Illuppallama_rolling_mean_7', 'Maha Illuppallama_rolling_std_7', 'Maha Illuppallama_rolling_median_7'
]
feature_check = all([col in X.columns for col in expected_features])

# Length Check for each target
vavuniya_check = df['Vavuniya'].between(0, 1000).all()
anuradhapura_check = df['Anuradhapura'].between(0, 1000).all()
maha_check = df['Maha Illuppallama'].between(0, 1000).all()

# Schema Check
expected_columns = len(expected_features)
def test_check_schema():
    actual_columns = X.shape[1]
    assert actual_columns == expected_columns, f"Expected {expected_columns} columns, but got {actual_columns}"
test_check_schema()

# Model Training and Evaluation
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
model_maha, mse_m, mae_m'''

# test.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

# Feature Check
expected_features = [
    'DayOfYear', 'Vavuniya_lag1', 'Anuradhapura_lag1', 'Maha Illuppallama_lag1',
    'Vavuniya_rolling_mean_7', 'Vavuniya_rolling_std_7', 'Vavuniya_rolling_median_7',
    'Anuradhapura_rolling_mean_7', 'Anuradhapura_rolling_std_7', 'Anuradhapura_rolling_median_7',
    'Maha Illuppallama_rolling_mean_7', 'Maha Illuppallama_rolling_std_7', 'Maha Illuppallama_rolling_median_7'
]
feature_check = all([col in X.columns for col in expected_features])

# Length Check for each target
vavuniya_check = df['Vavuniya'].between(0, 1000).all()  # Adjust based on realistic limits
anuradhapura_check = df['Anuradhapura'].between(0, 1000).all()  # Adjust based on realistic limits
maha_check = df['Maha Illuppallama'].between(0, 1000).all()  # Adjust based on realistic limits

# Schema Check
expected_columns = len(expected_features)

# realistic limits
def test_check_schema():
    actual_columns = X.shape[1]
    assert actual_columns == expected_columns, f"Expected {expected_columns} columns, but got {actual_columns}"

test_check_schema()



# Convert checks to human-readable results
feature_check_result = "Passed &#9989;" if feature_check else "Failed &#10540;"
vavuniya_check_result = "Passed &#9989;" if vavuniya_check else "Failed &#10540;"
anuradhapura_check_result = "Passed &#9989;" if anuradhapura_check else "Failed &#10540;"
maha_check_result = "Passed &#9989;" if maha_check else "Failed &#10540;"

# Save the results to test.txt
with open("test.txt", 'w') as outfile:
    outfile.write("Feature Test: %s\n" % feature_check_result)
    outfile.write("Vavuniya Length Test: %s\n" % vavuniya_check_result)
    outfile.write("Anuradhapura Length Test: %s\n" % anuradhapura_check_result)
    outfile.write("Maha Illuppallama Length Test: %s\n" % maha_check_result)

# Schema Check
expected_columns = 13
def test_check_schema():
    actual_columns = X.shape[1]
    # Check if the number of columns matches the expected number
    assert actual_columns == expected_columns, f"Expected {expected_columns} columns, got {actual_columns}"

try:
    test_check_schema()
    schema_check_result = "Passed &#9989;"
except AssertionError as e:
    schema_check_result = f"Failed &#10540; ({str(e)})"

# Save schema check result to test.txt
with open("test.txt", 'a') as outfile:
    outfile.write("Schema Test: %s\n" % schema_check_result)


