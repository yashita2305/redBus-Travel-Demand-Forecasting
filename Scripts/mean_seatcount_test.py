# Training random forest 
import pandas as pd
import numpy as np
import sklearn
train=pd.read_csv("/home/cair/Desktop/weed_detection_akhil/akhil_weed/redbus/train/train.csv")
train['doj_day'] = train['doj'].str.split('-').str[2]
train['doj_month'] = train['doj'].str.split('-').str[1]
train['doj_year'] = train['doj'].str.split('-').str[0]
train.drop('doj', axis=1, inplace=True)
train['doj_day'] = train['doj_day'].astype(int)
train['doj_month'] = train['doj_month'].astype(int)
train['doj_year'] = train['doj_year'].astype(int)
df=train.copy(deep=True)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib


# Force-renaming the date columns to expected names
df = df.rename(columns={
    'doj_day': 'day',
    'doj_month': 'month',
    'doj_year': 'year'
})

# Now convert to datetime using correct names
df['doj'] = pd.to_datetime(df[['year', 'month', 'day']])
df['doj_encoded'] = df['doj'].map(lambda x: x.toordinal())


# 🔹 Step 3: Feature and target selection
X = df[['srcid', 'destid', 'doj_encoded']]
y = df['final_seatcount']

# 🔹 Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 5: Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 350, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# 🔹 Step 6: GridSearchCV
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           scoring='neg_root_mean_squared_error',
                           cv=5,
                           n_jobs=-1,
                           verbose=3)

grid_search.fit(X_train, y_train)

# 🔹 Step 7: Get best params
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 🔹 Step 8: Train best model
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# 🔹 Step 9: Evaluate
y_pred = best_rf.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print("Test RMSE:", rmse)

# 🔹 Step 10: Save the model
joblib.dump(best_rf, "best_random_forest_regressor.pkl")
print("Model saved as 'best_random_forest_regressor.pkl'")
