def predict_seatcount(test_data, model_path='lgbm_model.pkl'):
    """
    Predict final_seatcount for test data and add predictions to the DataFrame.
    
    Args:
        test_data (pandas.DataFrame or str): Test data or path to CSV with columns
            ['doj', 'srcid', 'destid']
        model_path (str): Path to trained LightGBM model
        
    Returns:
        pandas.DataFrame: Test data with added 'final_seatcount' column
    """
    # Load test data if provided as path
    if isinstance(test_data, str):
        df = pd.read_csv(test_data)
    else:
        df = test_data.copy()
    
    # Store original columns for output
    original_cols = df.columns.tolist()
    
    # Preprocess test data
    X_test = preprocess_test_data(df)
    
    # Load trained model
    print("Loading trained model...")
    model = joblib.load(model_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Add predictions to DataFrame
    df['final_seatcount'] = predictions
    
    # Keep only original columns plus final_seatcount
    output_cols = original_cols + ['final_seatcount']
    df = df[output_cols]
    
    # Save predictions
    df.to_csv('test_data_with_predictions.csv', index=False)
    print("Predictions saved to 'test_data_with_predictions.csv'")
    
    return df

if __name__ == "__main__":
    # Example test data
    test_data = test.copy(deep=True)
    
    
    # Predict and add final_seatcount
    result = predict_seatcount(test_data)
    print("\nTest data with predictions:")
    print(result)