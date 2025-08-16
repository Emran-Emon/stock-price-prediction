import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ========== SETTINGS ==========
STOCKS = ["AAPL", "MSFT", "FB", "TSLA"]
DATA_FOLDER = "indiv-datasets"

# ========== LOAD & COMBINE TRAINING DATA ==========
train_dfs = []
for stock in STOCKS:
    train_path = os.path.join(DATA_FOLDER, f"{stock}_Train.csv")
    df_train = pd.read_csv(train_path)
    df_train["Stock"] = stock
    train_dfs.append(df_train)

train_df = pd.concat(train_dfs, ignore_index=True)

# Drop non-numeric columns for training
X_train = train_df.drop(columns=["Close", "Stock", "Date"])
y_train = train_df["Close"]

print(f"✅ Combined training set shape: {X_train.shape}")

# ========== TRAIN MODEL WITH GRIDSEARCH ==========
param_grid = {
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"✅ Best parameters: {grid_search.best_params_}")

# ========== EVALUATE EACH STOCK SEPARATELY ==========
def evaluate_stock(stock, model):
    test_path = os.path.join(DATA_FOLDER, f"{stock}_Test.csv")
    df_test = pd.read_csv(test_path)

    # Drop Date and Stock for prediction
    X_test = df_test.drop(columns=["Close", "Stock", "Date"])
    y_test = df_test["Close"]

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {stock} Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(y_test.values, label="Actual Prices", color="blue")
    plt.plot(y_pred, label="Predicted Prices", color="red", linestyle="--")
    plt.title(f"{stock} - Actual vs Predicted Prices")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run evaluation for each stock
for stock in STOCKS:
    evaluate_stock(stock, best_model)