import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import holidays
from datetime import timedelta

def preprocess_data(trans_path, train_path):
    """
    Preprocess transactions and train data to produce final dataset.
    
    Args:
        trans_path (str): Path to transactions CSV file
        train_path (str): Path to train CSV file
        
    Returns:
        pandas.DataFrame: Preprocessed dataset
    """
    # Read data
    trans = pd.read_csv(trans_path)
    train = pd.read_csv(train_path)
    
    # 1. Label Encoding for categorical columns
    le = LabelEncoder()
    categorical_cols = ['srcid_tier', 'destid_tier', 'srcid_region', 'destid_region']
    for col in categorical_cols:
        trans[f"{col}_encoded"] = le.fit_transform(trans[col])
        trans = trans.drop(col, axis=1)
    
    # 2. Date splitting for doj and doi
    def split_date(df, date_col):
        df[f"{date_col}_date"] = df[date_col].str.split('-').str[2].astype(int)
        df[f"{date_col}_month"] = df[date_col].str.split('-').str[1].astype(int)
        df[f"{date_col}_year"] = df[date_col].str.split('-').str[0].astype(int)
        return df.drop(date_col, axis=1)
    
    trans = split_date(trans, 'doj')
    trans = split_date(trans, 'doi')
    train = split_date(train, 'doj')
    
    # 3. Group and merge counts
    group_cols = [
        'srcid', 'destid',
        'srcid_tier_encoded', 'destid_tier_encoded',
        'srcid_region_encoded', 'destid_region_encoded',
        'doj_date', 'doj_month', 'doj_year'
    ]
    
    grouped_counts = trans.groupby(group_cols).size().reset_index(name='count')
    df_with_count = trans.merge(grouped_counts, on=group_cols, how='left')
    
    # 4. Merge with train data
    train_subset = train[["srcid", "destid", "doj_date", "doj_month", "doj_year", "final_seatcount"]]
    final_data = df_with_count.merge(train_subset, on=["srcid", "destid", "doj_date", "doj_month", "doj_year"], how="left")
    
    # 5. Calculate mean final seatcount
    final_data["mean_final_seatcount"] = final_data["final_seatcount"] / final_data["count"]
    
    # 6. Calculate seat to search ratio
    final_data["seat_to_search"] = final_data["cumsum_seatcount"] / final_data["cumsum_searchcount"]
    
    # 7. Fill missing seat_to_search values
    null_indices = final_data[final_data['seat_to_search'].isnull()].index
    for idx in null_indices:
        row = final_data.loc[idx]
        matching_rows = final_data[
            (final_data['dbd'] == row['dbd']) &
            (final_data['srcid'] == row['srcid']) &
            (final_data['destid'] == row['destid']) &
            (final_data['seat_to_search'].notnull())
        ]
        if not matching_rows.empty:
            final_data.at[idx, 'seat_to_search'] = matching_rows['seat_to_search'].median()
    
    # 8. Smart fill for mean_final_seatcount
    def smart_fill_mean_final_seatcount(df):
        for group_cols in [
            ['srcid', 'destid'],
            ['srcid_region_encoded', 'destid_region_encoded', 'doj_month'],
            ['srcid_tier_encoded', 'destid_tier_encoded'],
            ['doj_month']
        ]:
            df['mean_final_seatcount'] = df.groupby(group_cols)['mean_final_seatcount']\
                                          .transform(lambda x: x.fillna(x.mean()))
        df['mean_final_seatcount'] = df['mean_final_seatcount'].fillna(df['mean_final_seatcount'].mean())
        return df
    
    final_data = smart_fill_mean_final_seatcount(final_data)
    
    # 9. Holiday and long weekend features
    final_data['doj'] = pd.to_datetime(final_data[['doj_year', 'doj_month', 'doj_date']]
                                     .rename(columns={'doj_year': 'year', 'doj_month': 'month', 'doj_date': 'day'}))
    
    ind_holidays = holidays.India(years=range(2019, 2026))
    holiday_dates = set(ind_holidays.keys())
    holiday_buffered = set()
    for h in holiday_dates:
        holiday_buffered.update([h - timedelta(days=1), h, h + timedelta(days=1)])
    
    final_data['is_holiday'] = final_data['doj'].isin(holiday_buffered).astype(int)
    final_data['is_long_weekend'] = final_data.apply(
        lambda row: int(row['is_holiday'] == 1 and row['doj'].weekday() in [4, 5, 6]), axis=1)
    
    # 10. Day of week features
    final_data['doj_dayofweek'] = final_data['doj'].dt.dayofweek
    final_data['mean_seats_per_dow'] = final_data.groupby('doj_dayofweek')['mean_final_seatcount'].transform('mean')
    
    # 11. High demand route
    final_data['is_high_demand_route'] = (final_data['mean_final_seatcount'] > 100).astype(int)
    
    # 12. Log transformations
    final_data['log_cumsum_seatcount'] = np.log1p(final_data['cumsum_seatcount'])
    final_data['log_cumsum_searchcount'] = np.log1p(final_data['cumsum_searchcount'])
    
    # 13. School holidays and wedding season
    final_data['doj_day'] = final_data['doj'].dt.day
    final_data['doj_month'] = final_data['doj'].dt.month
    
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
    
    final_data['school_holidays'] = final_data.apply(is_school_holiday, axis=1)
    final_data['wedding_season'] = final_data.apply(is_wedding_season, axis=1)
    
    # 14. Final cleanup
    final_data = final_data.drop([
        'count', 'final_seatcount', 'doj', 'doj_day',
        'log_cumsum_seatcount', 'log_cumsum_searchcount'
    ], axis=1, errors='ignore')
    
    # Convert date columns to numeric
    cols_to_convert = ['doj_date', 'doj_year', 'doi_date', 'doi_month', 'doi_year']
    final_data[cols_to_convert] = final_data[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    return final_data

# Example usage:
# final_data = preprocess_data("/home/cair/Desktop/weed_detection_akhil/akhil_weed/redbus/train/transactions.csv", "/home/cair/Desktop/weed_detection_akhil/akhil_weed/redbus/train/train.csv")
# final_data.to_csv("preprocessed_data.csv", index=False)