import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# -----------------------------------------
# Load CSV
# -----------------------------------------
file_path = r"E:/airbnb data/AB_NYC_2019.csv"   # CHANGE PATH IF NEEDED
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print(df.head())


# -----------------------------------------
# Column selection based on REAL CSV columns
# (Your dataset has only these columns available)
# -----------------------------------------

valid_columns = [
    'id',
    'name',
    'host_id',
    'host_name',
    'neighbourhood_group',
    'neighbourhood',
    'latitude',
    'longitude',
    'room_type',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'last_review',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'
]

df = df[valid_columns].copy()


# -----------------------------------------
# Cleaning & Preprocessing
# -----------------------------------------

# Handle missing values
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Remove rows with price 0 or extremely high outliers
df = df[df['price'] > 0]

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['price'] >= lower) & (df['price'] <= upper)]

print("After outlier removal:", df.shape)


# -----------------------------------------
# Encoding Categorical Columns
# -----------------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=['neighbourhood_group', 'neighbourhood', 'room_type'],
    drop_first=True
)


# -----------------------------------------
# Prepare Features and Target
# -----------------------------------------

X = df_encoded.drop(columns=['price', 'name', 'host_name', 'last_review'])
y = df_encoded['price']

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)


# -----------------------------------------
# Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------------------
# Linear Regression Model
# -----------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("\n--- Linear Regression ---")
print("MAE :", round(mean_absolute_error(y_test, pred_lr), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_lr)), 2))
print("R²  :", round(r2_score(y_test, pred_lr), 4))


# -----------------------------------------
# XGBoost Model
# -----------------------------------------
xgb = XGBRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)

print("\n--- XGBoost ---")
print("MAE :", round(mean_absolute_error(y_test, pred_xgb), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_xgb)), 2))
print("R²  :", round(r2_score(y_test, pred_xgb), 4))


# -----------------------------------------
# Feature Importance
# -----------------------------------------
importances = xgb.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

print("\nTop 20 Important Features:")
for i in sorted_idx[:20]:
    print(f"{features[i]}: {importances[i]:.4f}")


# -----------------------------------------
# Example Prediction Function
# -----------------------------------------
def predict_price(model, example_dict):
    df_example = pd.DataFrame([example_dict]).reindex(columns=X.columns, fill_value=0)
    return model.predict(df_example)[0]


# Example usage
example = X_train.iloc[0].to_dict()
print("\nExample predicted price:", predict_price(xgb, example))
