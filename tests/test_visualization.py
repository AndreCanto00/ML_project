import pytest
import pandas as pd
import numpy as np
from src.visualization import (
    plot_categorical_by_target,
    plot_numerical_distributions,
    plot_numerical_by_target,
    plot_feature_relationships
)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'cat1': [1, 2, 1, 2, 1],
        'cat2': [0, 1, 0, 1, 0],
        'num1': [1.5, 2.5, 3.5, 4.5, 5.5],
        'num2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'failure': [0, 1, 1, 0, 1]
    })

def test_plot_categorical_by_target(sample_data):
    # Test that function runs without errors
    categorical_cols = ['cat1', 'cat2']
    plot_categorical_by_target(sample_data, categorical_cols)
    plt.close('all')

def test_plot_numerical_distributions(sample_data):
    # Test that function runs without errors
    numerical_cols = ['num1', 'num2']
    plot_numerical_distributions(sample_data, numerical_cols)
    plt.close('all')

def test_plot_numerical_by_target(sample_data):
    # Test that function runs without errors
    numerical_cols = ['num1', 'num2']
    plot_numerical_by_target(sample_data, numerical_cols)
    plt.close('all')

def test_plot_feature_relationships(sample_data):
    # Test that function runs without errors
    columns = ['num1', 'num2']
    plot_feature_relationships(sample_data, columns)
    plt.close('all')