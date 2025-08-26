import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from lightgbm import LGBMRegressor
import joblib

# === Step 1: Load Data ===
data = pd.read_csv('/home/cair/Desktop/weed_detection_akhil/akhil_weed/redbus/train/train.csv')
df = pd.DataFrame(data)

# Extract day, month, year
df['doj_day'] = df['doj'].str.split('-').str[2].astype(int)
df['doj_month'] = df['doj'].str.split('-').str[1].astype(int)
df['doj_year'] = df['doj'].str.split('-').str[0].astype(int)

df = df.rename(columns={
    'doj_day': 'day',
    'doj_month': 'month',
    'doj_year': 'year'
})

# Create datetime & ordinal version
df['doj'] = pd.to_datetime(df[['year', 'month', 'day']])
df['doj_encoded'] = df['doj'].map(lambda x: x.toordinal())

# === Step 2: Split into features and target ===
X = df.drop(columns=['final_seatcount', 'doj'])  # ✅ Exclude datetime column
y = df['final_seatcount']

# === Step 3: LightGBM model and hyperparameter grid ===
lgbm = LGBMRegressor(device='gpu')

param_dist = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.001, 0.01, 0.05],
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [3, 5, 7, -1],
    'boosting_type': ['gbdt', 'dart'],
    'min_child_samples': [5, 10, 20]
}

rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)

# === Step 4: RandomizedSearchCV ===
print("🔍 Performing RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    scoring=rmse_scorer,
    n_iter=20,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

print("✅ Best Parameters:")
print(random_search.best_params_)

# === Step 5: Evaluate best model ===
best_lgbm = random_search.best_estimator_
cv_rmse = -cross_val_score(best_lgbm, X, y, scoring='neg_root_mean_squared_error', cv=5)
print(f"📊 Cross-Validated RMSE Scores: {cv_rmse}")
print(f"✅ Mean CV RMSE: {np.mean(cv_rmse):.4f}")

# === Step 6: Retrain on full data ===
best_lgbm.fit(X, y)

# === Step 7: Save model ===
model_path = "lgbm_best_model.pkl"
joblib.dump(best_lgbm, model_path)
print(f"💾 Model saved as: {model_path}")

# === Step 8: Load model for inference ===
loaded_model = joblib.load(model_path)
print("🔄 Model loaded successfully for inference.")
