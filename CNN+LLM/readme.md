# Chest X-ray Analysis

A tool for analyzing chest X-ray images using deep learning techniques.

## Overview

This project provides a framework for processing and analyzing chest X-ray images, including:

- Loading and preprocessing medical imaging data
- Generating embeddings for both images and text reports
- Analyzing the normalcy of chest X-rays
- Visualizing the results

## Dataset

The dataset contains chest X-ray images along with:
- Medical captions and findings
- Clinical impressions
- Normalcy classifications (Normal/Abnormal)

## Installation

git clone https://github.com/username/chest-xray-analysis.git
cd chest-xray-analysis
pip install -r requirements.txt


## Usage

python main.py --data_path /path/to/dataset --output_path ./output --visualize


### Arguments

- `--data_path`: Path to the dataset directory (required)
- `--output_path`: Path to save results (default: ./output)
- `--visualize`: Generate visualizations

## Project Structure

chest-xray-analysis/
│
├── main.py # Entry point for the application
├── README.md # Project documentation
├── requirements.txt # Dependencies
│
├── src/ # Source code
│ ├── data/ # Data handling
│ │ ├── loader.py # Data loading functions
│ │ └── processor.py # Data preprocessing functions
│ │
│ ├── models/ # Model definitions
│ │ └── embeddings.py # Embedding models
│ │
│ └── utils/ # Utility functions
│ └── visualization.py # Visualization utilities
│
└── tests/ # Unit tests


## License

MIT
