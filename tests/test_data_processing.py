import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_data, check_duplicates, analyze_missing_values, clean_dataset

def test_load_data():
    url = "https://raw.githubusercontent.com/AndreCanto00/ML_project/main/tyres_train.csv"
    data = load_data(url)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_check_duplicates():
    # Create sample data with duplicates
    data = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6]
    })
    duplicates = check_duplicates(data)
    assert len(duplicates) == 1
    assert duplicates.iloc[0].tolist() == [2, 5]

def test_analyze_missing_values():
    # Create sample data with missing values
    data = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, 6],
        'C': [np.nan, np.nan, np.nan]
    })
    has_missing, missing_count = analyze_missing_values(data)
    
    assert has_missing['A'] == True
    assert has_missing['B'] == False
    assert has_missing['C'] == True
    assert missing_count['A'] == 1
    assert missing_count['B'] == 0
    assert missing_count['C'] == 3

def test_clean_dataset():
    # Create sample data with a column full of missing values
    data = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, 6],
        'C': [np.nan, np.nan, np.nan]
    })
    cleaned_data = clean_dataset(data)
    
    assert 'C' not in cleaned_data.columns
    assert len(cleaned_data.columns) == 2