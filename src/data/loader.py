#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for loading and managing the chest X-ray dataset.
"""

import os
import pandas as pd
from typing import Dict

def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the chest X-ray dataset from the specified path.
    
    Args:
        data_path (str): Path to the dataset directory
    
    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """
    # Construct path to the main CSV file
    csv_path = os.path.join(data_path, 'metadata.csv')
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Add full paths to images
    df['imgs'] = df['imgs'].apply(lambda x: os.path.join(data_path, x))
    
    # Filter out rows with missing values if needed
    df = df.dropna(subset=['imgs', 'captions', 'findings', 'impression'])
    
    print(f"Loaded dataset with {len(df)} entries")
    return df

def split_dataset(df: pd.DataFrame, 
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the dataset
        test_size (float, optional): Proportion for test split
        val_size (float, optional): Proportion for validation split
        random_state (int, optional): Random seed
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with train, val, and test DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Then split train_val into train and validation sets
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=random_state
    )
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
