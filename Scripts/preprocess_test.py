import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import holidays
from datetime import timedelta

def preprocess_prediction_data(pred_data, trans_path):
    """
    Preprocess prediction data to match the features of the trained model's input.
    Sets dbd to 15 days and calculates doi accordingly.
    
    Args:
        pred_data (pandas.DataFrame or str): Prediction data or path to CSV with columns
            ['route_key', 'doj', 'srcid', 'destid']
        trans_path (str): Path to transactions CSV file for encoded features
        
    Returns:
        pandas.DataFrame: Preprocessed prediction data ready for model prediction
    """
    # Read prediction data if provided as path
    if isinstance(pred_data, str):
        df = pd.read_csv(pred_data)
    else:
        df = pred_data.copy()
    
    # Read transactions data for encoded features
    trans = pd.read_csv(trans_path)
    
    # 1. Extract date components from doj
    df['doj'] = pd.to_datetime(df['doj'], format='%m/%d/%Y')
    df['doj_date'] = df['doj'].dt.day.astype(int)
    df['doj_month'] = df['doj'].dt.month.astype(int)
    df['doj_year'] = df['doj'].dt.year.astype(int)
    
    # 2. Set dbd to 15 and derive doi
    df['dbd'] = 15
    df['doi'] = df['doj'] - pd.to_timedelta(df['dbd'], unit='D')
    df['doi_date'] = df['doi'].dt.day.astype(int)
    df['doi_month'] = df['doi'].dt.month.astype(int)
    df['doi_year'] = df['doi'].dt.year.astype(int)
    
    # 3. Get encoded features from transactions data
    # Create lookup tables for encoded features
    srcid_tier_map = trans[['srcid', 'srcid_tier']].drop_duplicates().set_index('srcid')
    destid_tier_map = trans[['destid', 'destid_tier']].drop_duplicates().set_index('destid')
    srcid_region_map = trans[['srcid', 'srcid_region']].drop_duplicates().set_index('srcid')
    destid_region_map = trans[['destid', 'destid_region']].drop_duplicates().set_index('destid')
    
    # Label encode categorical columns
    le = LabelEncoder()
    
    # Encode srcid_tier
    srcid_tier_map['srcid_tier_encoded'] = le.fit_transform(srcid_tier_map['srcid_tier'])
    df = df.merge(srcid_tier_map[['srcid_tier_encoded']], left_on='srcid', right_index=True, how='left')
    
    # Encode destid_tier
    destid_tier_map['destid_tier_encoded'] = le.fit_transform(destid_tier_map['destid_tier'])
    df = df.merge(destid_tier_map[['destid_tier_encoded']], left_on='destid', right_index=True, how='left')
    
    # Encode srcid_region
    srcid_region_map['srcid_region_encoded'] = le.fit_transform(srcid_region_map['srcid_region'])
    df = df.merge(srcid_region_map[['srcid_region_encoded']], left_on='srcid', right_index=True, how='left')
    
    # Encode destid_region
    destid_region_map['destid_region_encoded'] = le.fit_transform(destid_region_map['destid_region'])
    df = df.merge(destid_region_map[['destid_region_encoded']], left_on='destid', right_index=True, how='left')
    
    # Fill missing encoded values with mode
    for col in ['srcid_tier_encoded', 'destid_tier_encoded', 'srcid_region_encoded', 'destid_region_encoded']:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
    
    # 4. Holiday and long weekend features
    ind_holidays = holidays.India(years=range(2019, 2026))
    holiday_dates = set(ind_holidays.keys())
    holiday_buffered = set()
    for h in holiday_dates:
        holiday_buffered.update([h - timedelta(days=1), h, h + timedelta(days=1)])
    
    df['is_holiday'] = df['doj'].isin(holiday_buffered).astype(int)
    df['is_long_weekend'] = df.apply(
        lambda row: int(row['is_holiday'] == 1 and row['doj'].weekday() in [4, 5, 6]), axis=1)
    
    # 5. Day of week features
    df['doj_dayofweek'] = df['doj'].dt.dayofweek
    
    # Estimate mean_seats_per_dow from transactions data
    trans['doj'] = pd.to_datetime(trans[['doj_year', 'doj_month', 'doj_date']]
                                 .rename(columns={'doj_year': 'year', 'doj_month': 'month', 'doj_date': 'day'}))
    trans['doj_dayofweek'] = trans['doj'].dt.dayofweek
    dow_means = trans.groupby('doj_dayofweek')['mean_final_seatcount'].mean()
    df['mean_seats_per_dow'] = df['doj_dayofweek'].map(dow_means).fillna(dow_means.mean())
    
    # 6. High demand route (estimate from transactions data)
    route_means = trans.groupby(['srcid', 'destid'])['mean_final_seatcount'].mean()
    df['mean_final_seatcount'] = df.set_index(['srcid', 'destid']).index.map(route_means)
    df['is_high_demand_route'] = (df['mean_final_seatcount'] > 100).astype(int)
    df['mean_final_seatcount'] = df['mean_final_seatcount'].fillna(route_means.mean())
    
    # 7. School holidays and wedding season
    df['doj_day'] = df['doj'].dt.day
    df['doj_month'] = df['doj'].dt.month
    
    def is_school_holiday(row):
        m, d = row['doj_month'], row['doj_day']
        return int(
            (m == 5 and d >= 20) or
            (m == 6) or
            (m == 7 and d <= 20) or
            (m == 12 and d >= 20) or
            (m == 1 and d <= 20)
        )
    
    def is_wedding_season(row):
        m, d = row['doj_month'], row['doj_day']
        return int(
            (m == 1 and d >= 15) or
            (m == 2) or
            (m == 3 and d <= 15) or
            (m == 5 and d >= 15) or
            (m == 6) or
            (m == 7 and d <= 15) or
            (m == 11 and d >= 15) or
            (m == 12 and d <= 15)
        )
    
    df['school_holidays'] = df.apply(is_school_holiday, axis=1)
    df['wedding_season'] = df.apply(is_wedding_season, axis=1)
    
    # 8. Add cumsum_searchcount and seat_to_search
    median_features = trans.groupby(['srcid', 'destid'])[['cumsum_searchcount', 'seat_to_search']].median()
    df = df.merge(median_features, left_on=['srcid', 'destid'], right_index=True, how='left')
    
    # Fill remaining missing values
    for col in ['cumsum_searchcount', 'seat_to_search']:
        df[col] = df[col].fillna(trans[col].median())
    
    # 9. Drop unnecessary columns
    df = df.drop(['route_key', 'doj', 'doi', 'doj_day'], axis=1, errors='ignore')
    
    # 10. Ensure correct data types
    cols_to_convert = ['doj_date', 'doj_year', 'doi_date', 'doi_month', 'doi_year']
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # 11. Reorder columns to match final_data (excluding cumsum_seatcount)
    expected_columns = [
        'srcid', 'destid', 'dbd', 'cumsum_searchcount', 'srcid_tier_encoded',
        'destid_tier_encoded', 'srcid_region_encoded', 'destid_region_encoded',
        'doj_date', 'doj_month', 'doj_year', 'doi_date', 'doi_month', 'doi_year',
        'mean_final_seatcount', 'seat_to_search', 'is_holiday', 'is_long_weekend',
        'doj_dayofweek', 'mean_seats_per_dow', 'is_high_demand_route',
        'school_holidays', 'wedding_season'
    ]
    df = df[expected_columns]
    
    return df

# Example usage:
# pred_data = pd.read_csv("prediction_data.csv")
# preprocessed_pred = preprocess_prediction_data(pred_data, "/path/to/transactions.csv")