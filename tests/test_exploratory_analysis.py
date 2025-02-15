import pytest
import pandas as pd
import numpy as np
from src.exploratory_analysis import (
    analyze_target_distribution,
    split_features_by_type,
    analyze_feature_types
)

def test_analyze_target_distribution():
    # Create sample data
    data = pd.DataFrame({
        'failure': [0, 1, 1, 1, 0],
        'feature': [1, 2, 3, 4, 5]
    })
    
    result = analyze_target_distribution(data)
    assert 'distribution' in result
    assert 'percentages' in result
    assert result['distribution'][0] == 2
    assert result['distribution'][1] == 3
    assert result['percentages'][0] == 40.0
    assert result['percentages'][1] == 60.0

def test_split_features_by_type():
    # Create sample data with mixed types
    data = pd.DataFrame({
        'failure': [0, 1, 1],
        'cat1': [1, 2, 3],
        'cat2': [0, 1, 0],
        'num1': [1.5, 2.5, 3.5],
        'num2': [0.1, 0.2, 0.3]
    })
    
    categorical, numerical = split_features_by_type(
        data,
        categorical_exclude=['cat2'],
        numerical_exclude=['num2']
    )
    
    assert 'cat1' in categorical.columns
    assert 'cat2' not in categorical.columns
    assert 'num1' in numerical.columns
    assert 'num2' not in numerical.columns
    assert 'failure' not in categorical.columns
    assert 'failure' not in numerical.columns

def test_analyze_feature_types():
    # Create sample data with different types
    data = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.0, 2.0, 3.0],
        'str_col': ['a', 'b', 'c']
    })
    
    dtypes = analyze_feature_types(data)
    
    assert dtypes['int_col'] == np.int64
    assert dtypes['float_col'] == np.float64
    assert dtypes['str_col'] == object