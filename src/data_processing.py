import pandas as pd
from typing import Tuple

def load_data(url: str) -> pd.DataFrame:
    """
    Load data from a CSV URL.
    
    Args:
        url (str): URL of the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(url)

def check_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check and return duplicated rows in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Duplicated rows if any
    """
    return data[data.duplicated()]

def analyze_missing_values(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Analyze missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        Tuple[pd.Series, pd.Series]: 
            - Boolean series indicating columns with missing values
            - Count of missing values per column
    """
    return data.isna().any(), data.isna().sum()

def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing columns with too many missing values.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset with removed columns
    """
    return data.dropna(axis=1)