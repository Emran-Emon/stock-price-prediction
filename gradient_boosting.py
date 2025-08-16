import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===== STEP 0: Config =====
data_folder = "indiv-datasets"
stocks = ["AAPL", "FB", "MSFT", "TSLA"]

# ===== FUNCTION TO ADD LAG & MOVING AVG FEATURES =====
def add_lag_features(df, target_col="Close", lags=[1,2,3], ma_windows=[3,5]):
    df = df.sort_values("Date").copy()
    
    # Lag features
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    
    # Moving averages
    for window in ma_windows:
        df[f"{target_col}_ma{window}"] = df[target_col].rolling(window).mean()
    
    df = df.dropna().reset_index(drop=True)
    return df

# ===== STEP 1: Loop through stocks =====
for stock in stocks:
    print(f"\n===== Training for {stock} =====")
    
    # Load CSVs
    train = pd.read_csv(os.path.join(data_folder, f"{stock}_Train.csv"))
    val = pd.read_csv(os.path.join(data_folder, f"{stock}_Validation.csv"))
    test = pd.read_csv(os.path.join(data_folder, f"{stock}_Test.csv"))
    
    # ===== STEP 2: Add lag & moving average features =====
    train = add_lag_features(train)
    val = add_lag_features(val)
    test = add_lag_features(test)
    
    # Features & target
    feature_cols = ["Open", "High", "Low", "Volume"] + \
                   [col for col in train.columns if "lag" in col or "ma" in col]
    target_col = "Close"
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    # ===== STEP 3: Gradient Boosting + GridSearch =====
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print("Best Parameters:", grid_search.best_params_)
    print("Validation R²:", best_model.score(X_val, y_val))
    
    # ===== STEP 4: Predictions =====
    y_pred = best_model.predict(X_test)
    
    # ===== STEP 5: Evaluation Metrics =====
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{stock} Test Metrics:")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    
    # ===== STEP 6: Visualization =====
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual Price", color="blue")
    plt.plot(y_pred, label="Predicted Price", color="red")
    plt.title(f"{stock} Actual vs Predicted Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()