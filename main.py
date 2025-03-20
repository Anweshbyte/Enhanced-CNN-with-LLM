#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the Chest X-ray Analysis application.
"""

import argparse
import os
from src.data.loader import load_dataset
from src.data.processor import preprocess_data
from src.models.embeddings import generate_embeddings
from src.utils.visualization import visualize_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Chest X-ray Analysis Tool')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Path to save results (default: ./output)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    return parser.parse_args()

def main():
    """Main function to run the analysis pipeline."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    data_df = load_dataset(args.data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_df = preprocess_data(data_df)
    
    # Generate embeddings
    print("Generating embeddings...")
    results_df = generate_embeddings(processed_df)
    
    # Save results
    results_path = os.path.join(args.output_path, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        visualize_results(results_df, args.output_path)
        print(f"Visualizations saved to {args.output_path}")

if __name__ == "__main__":
    main()
