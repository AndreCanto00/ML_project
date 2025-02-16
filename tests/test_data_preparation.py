import pytest
import pandas as pd
import numpy as np
import os
from src.data_preparation import (
    standardize_features,
    plot_scaled_distributions,
    compute_correlations,
    combine_features
)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'num1': [1.0, 2.0, 3.0],
        'num2': [4.0, 5.0, 6.0],
        'failure': [0, 1, 1]
    })

def test_standardize_features(sample_data, tmp_path):
    # Use temporary path for scaler
    scaler_path = tmp_path / "test_scaler.pkl"
    
    scaled_data, scaler = standardize_features(
        sample_data, 
        scaler_path=str(scaler_path)
    )
    
    # Check if scaler was saved
    assert os.path.exists(scaler_path)
    
    # Check if data was standardized correctly
    assert scaled_data.shape == (3, 2)  # Excluding target
    assert np.allclose(scaled_data.mean(), 0, atol=1e-10)
    assert np.allclose(scaled_data.std(), 1, atol=1e-10)

def test_compute_correlations(sample_data):
    corr = compute_correlations(sample_data)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (3, 3)  # 3x3 correlation matrix
    assert np.allclose(np.diag(corr), 1)  # Diagonal should be 1

def test_combine_features():
    cat_data = pd.DataFrame({
        'cat1': [1, 2, 3],
        'cat2': [0, 1, 0]
    })
    num_data = pd.DataFrame({
        'num1': [0.1, 0.2, 0.3],
        'num2': [0.4, 0.5, 0.6]
    })
    target = pd.Series([0, 1, 1], name='failure')
    
    combined = combine_features(cat_data, num_data, target)
    
    assert combined.shape == (3, 5)  # 2 cat + 2 num + 1 target
    assert list(combined.columns) == ['cat1', 'cat2', 'num1', 'num2', 'failure']