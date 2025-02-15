import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def analyze_target_distribution(data: pd.DataFrame, target_col: str = 'failure') -> Dict[int, int]:
    """
    Analyze the distribution of the target variable.
    
    Args:
        data (pd.DataFrame): Input dataset
        target_col (str): Name of the target column
        
    Returns:
        Dict[int, int]: Distribution of target classes
    """
    distribution = data.groupby(target_col).size()
    total = len(data)
    
    # Calculate percentages
    percentages = (distribution / total * 100).round(1)
    
    # Create visualization
    plt.style.use('ggplot')
    sns.countplot(y=data[target_col], data=data)
    plt.xlabel("count of each class")
    plt.ylabel("classes")
    plt.title(f"Distribution of {target_col}")
    plt.show()
    
    return {
        'distribution': distribution,
        'percentages': percentages
    }

def split_features_by_type(data: pd.DataFrame, 
                          target_col: str = 'failure',
                          categorical_exclude: list = None,
                          numerical_exclude: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split features into categorical and numerical DataFrames.
    
    Args:
        data (pd.DataFrame): Input dataset
        target_col (str): Name of target column to exclude
        categorical_exclude (list): Additional categorical columns to exclude
        numerical_exclude (list): Additional numerical columns to exclude
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Categorical and numerical features
    """
    if categorical_exclude is None:
        categorical_exclude = []
    if numerical_exclude is None:
        numerical_exclude = []
        
    # Get categorical features (integers in this case)
    categorical = data.select_dtypes(include=['int'])
    categorical = categorical.drop([target_col] + categorical_exclude, axis=1, errors='ignore')
    
    # Get numerical features
    numerical = data.drop(
        [target_col] + 
        list(categorical.columns) + 
        numerical_exclude, 
        axis=1, 
        errors='ignore'
    )
    
    return categorical, numerical

def analyze_feature_types(data: pd.DataFrame) -> pd.Series:
    """
    Analyze the data types of all features.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.Series: Data types of each column
    """
    return data.dtypes