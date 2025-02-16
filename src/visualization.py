import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_categorical_by_target(data: pd.DataFrame, 
                             categorical_cols: List[str], 
                             target_col: str = 'failure') -> None:
    """
    Plot histograms of categorical variables split by target value.
    
    Args:
        data (pd.DataFrame): Input dataset
        categorical_cols (List[str]): List of categorical columns to plot
        target_col (str): Name of the target column
    """
    # Split data by target
    data_0 = data[data[target_col] == 0]
    data_1 = data[data[target_col] == 1]
    
    # Create subplots
    n_cols = min(4, len(categorical_cols))
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=[15, 5*n_rows])
    axes = axes.flatten()
    fig.tight_layout(h_pad=10)
    
    for i, col in enumerate(categorical_cols):
        plt.sca(axes[i])
        plt.hist([data_0[col], data_1[col]], density=True)
        plt.xticks(rotation=90)
        plt.title(col)
        axes[i].legend(('0', '1'), loc='upper right')
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.show()

def plot_numerical_distributions(data: pd.DataFrame, 
                               numerical_cols: List[str]) -> None:
    """
    Plot histograms of numerical variables.
    
    Args:
        data (pd.DataFrame): Input dataset
        numerical_cols (List[str]): List of numerical columns to plot
    """
    n_cols = min(4, len(numerical_cols))
    n_rows = int(np.ceil(len(numerical_cols) / n_cols))
    
    data[numerical_cols].hist(layout=(n_rows, n_cols), 
                            figsize=(15, 5*n_rows))
    plt.tight_layout()
    plt.show()

def plot_numerical_by_target(data: pd.DataFrame, 
                           numerical_cols: List[str], 
                           target_col: str = 'failure') -> None:
    """
    Plot density distributions of numerical variables split by target value.
    
    Args:
        data (pd.DataFrame): Input dataset
        numerical_cols (List[str]): List of numerical columns to plot
        target_col (str): Name of the target column
    """
    # Split data by target
    X0 = data[data[target_col] == 0]
    X1 = data[data[target_col] == 1]
    
    n_cols = min(4, len(numerical_cols))
    n_rows = int(np.ceil(len(numerical_cols) / n_cols))
    
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, 
                            figsize=(5*n_cols, 5*n_rows))
    fig.tight_layout()
    
    for i, (ax, col) in enumerate(zip(axes.flat, numerical_cols)):
        sns.histplot(X0[col], color="blue", ax=ax, 
                    stat='density', element="step", alpha=0.3)
        sns.histplot(X1[col], color="red", ax=ax,
                    stat='density', element="step", alpha=0.3)
        ax.set_title(col)
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes.flat)):
        axes.flat[j].set_visible(False)
    
    plt.show()

def plot_feature_relationships(data: pd.DataFrame, 
                             feature_columns: list = None,
                             target_col: str = 'failure') -> None:
    """
    Create a pairplot of selected features colored by target value.
    
    Args:
        data (pd.DataFrame): Input dataset with both features and target column
        feature_columns (list, optional): List of feature columns to plot
        target_col (str): Name of the target column
    """
    if feature_columns is not None:
        plot_data = data[feature_columns + [target_col]]
    else:
        plot_data = data
        
    sns.pairplot(plot_data, hue=target_col)
    plt.show()