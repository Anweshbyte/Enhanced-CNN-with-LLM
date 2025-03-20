#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for visualizing the chest X-ray dataset and analysis results.
"""

import pandas as pd
import os
from typing import Optional

def visualize_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Generate visualizations for the dataset and analysis results.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset and results
        output_path (str): Path to save the visualizations
    """
    # Make sure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate distribution plots
    plot_class_distribution(df, output_path)
    
    # Generate sample visualizations
    visualize_samples(df, output_path)

def plot_class_distribution(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot the distribution of normal vs abnormal cases.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset
        output_path (str): Path to save the plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Count normal vs abnormal cases
        normalcy_counts = df['normalcy'].value_counts()
        
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=normalcy_counts.index, y=normalcy_counts.values)
        plt.title('Distribution of Normal vs Abnormal Cases')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_path, 'class_distribution.png'))
        plt.close()
        
    except ImportError:
        print("Warning: matplotlib or seaborn not installed. Skipping visualization.")

def visualize_samples(df: pd.DataFrame, output_path: str, num_samples: int = 5) -> None:
    """
    Visualize sample images with their findings and impressions.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset
        output_path (str): Path to save the visualizations
        num_samples (int, optional): Number of samples to visualize
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # Create samples directory
        samples_dir = os.path.join(output_path, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        # Get random samples for normal and abnormal cases
        normal_samples = df[df['normalcy'] == 'Normal'].sample(min(num_samples, len(df[df['normalcy'] == 'Normal'])))
        abnormal_samples = df[df['normalcy'] == 'Abnormal'].sample(min(num_samples, len(df[df['normalcy'] == 'Abnormal'])))
        
        # Visualize the samples
        for i, (_, row) in enumerate(normal_samples.iterrows()):
            visualize_single_sample(row, os.path.join(samples_dir, f'normal_sample_{i}.png'))
        
        for i, (_, row) in enumerate(abnormal_samples.iterrows()):
            visualize_single_sample(row, os.path.join(samples_dir, f'abnormal_sample_{i}.png'))
        
    except ImportError:
        print("Warning: matplotlib or Pillow not installed. Skipping visualization.")

def visualize_single_sample(row: pd.Series, output_path: str) -> None:
    """
    Visualize a single sample with its findings and impression.
    
    Args:
        row (pd.Series): Row from the dataset
        output_path (str): Path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Load and display the image
        try:
            img = Image.open(row['imgs'])
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image: {e}", 
                   horizontalalignment='center', verticalalignment='center')
        
        # Add the findings and impression as text
        findings = row.get('findings', 'N/A')
        impression = row.get('impression', 'N/A')
        normalcy = row.get('normalcy', 'N/A')
        
        fig.text(0.1, 0.05, f"Findings: {findings}", wrap=True, fontsize=10)
        fig.text(0.1, 0.02, f"Impression: {impression}", wrap=True, fontsize=10)
        fig.text(0.1, 0.95, f"Classification: {normalcy}", fontsize=12, 
                color='red' if normalcy == 'Abnormal' else 'green')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing sample: {e}")
