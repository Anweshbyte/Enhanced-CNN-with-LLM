#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for processing and transforming the chest X-ray dataset.
"""

import pandas as pd
import re

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the dataset
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Create structured captions if they don't exist
    if 'structured_caption' not in processed_df.columns:
        processed_df['structured_caption'] = processed_df.apply(
            lambda row: create_structured_caption(row), axis=1
        )
    
    # Replace [UNK] tokens with a more standard token
    text_columns = ['captions', 'findings', 'impression', 'structured_caption']
    for col in text_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].str.replace(r'\[UNK\]', '<UNK>', regex=True)
    
    return processed_df

def create_structured_caption(row: pd.Series) -> str:
    """
    Create a structured caption from findings and impression.
    
    Args:
        row (pd.Series): Row from the dataset
    
    Returns:
        str: Structured caption
    """
    findings = row.get('findings', '')
    impression = row.get('impression', '')
    
    structured_caption = f"[FINDINGS] {findings}"
    if impression:
        structured_caption += f" [IMPRESSION] {impression}"
    
    return structured_caption

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except those that are medically relevant
    text = re.sub(r'[^\w\s.,;:()/\-+]', '', text)
    
    return text.strip()
