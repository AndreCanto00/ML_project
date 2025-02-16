import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def standardize_features(data: pd.DataFrame, 
                        target_col: str = 'failure',
                        scaler_path: str = 'scaler.pkl') -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numerical features and save the scaler.
    
    Args:
        data (pd.DataFrame): Input dataset with numerical features
        target_col (str): Name of target column to exclude from standardization
        scaler_path (str): Path to save the scaler
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Standardized data and fitted scaler
    """
    # Separate features from target
    features = data.drop(target_col, axis=1, errors='ignore')
    
    # Fit and transform the data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )
    
    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return scaled_data, scaler

def plot_scaled_distributions(scaled_data: pd.DataFrame) -> None:
    """
    Plot boxplots of scaled features.
    
    Args:
        scaled_data (pd.DataFrame): Standardized data
    """
    plt.figure(figsize=(12, 6))
    scaled_data.boxplot(rot=90)
    plt.title('Distribution of Standardized Features')
    plt.tight_layout()
    plt.show()

def compute_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and visualize correlation matrix.
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1,
                center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr

def combine_features(categorical_data: pd.DataFrame, 
                    numerical_data: pd.DataFrame,
                    target_data: pd.Series) -> pd.DataFrame:
    """
    Combine categorical and numerical features with target.
    
    Args:
        categorical_data (pd.DataFrame): Categorical features
        numerical_data (pd.DataFrame): Numerical features
        target_data (pd.Series): Target variable
        
    Returns:
        pd.DataFrame: Combined dataset
    """
    return pd.concat(
        [categorical_data, numerical_data, target_data], 
        axis=1
    )