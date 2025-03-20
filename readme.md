# Chest X-ray Analysis

A tool for analyzing chest X-ray images using deep learning techniques.

## Overview

This project provides a framework for processing and analyzing chest X-ray images, including:

- Loading and preprocessing medical imaging data
- Generating embeddings for both images and text reports
- Analyzing the normalcy of chest X-rays
- Visualizing the results

## Dataset

@article{,
  title= {Indiana University - Chest X-Rays (PNG Images)},
  keywords= {radiology, chest x-ray},
  author= {OpenI},
  abstract= {1000 radiology reports for the chest x-ray images from the Indiana University hospital network.

To identify images associated with the reports, use XML tag. More than one image could be associated with a report.

![Example Image](https://i.imgur.com/5uR5snH.png)
  },
  license= {Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License},
  url= {https://openi.nlm.nih.gov/faq.php}
}

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

```plaintext
chest-xray-analysis/
│
├── main.py                 # Entry point for the application
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
│
├── src/                    # Source code
│   ├── data/               # Data handling
│   │   ├── loader.py       # Data loading functions
│   │   └── processor.py    # Data preprocessing functions
│   │
│   ├── models/             # Model definitions
│   │   └── embeddings.py   # Embedding models
│   │
│   └── utils/              # Utility functions
│       └── visualization.py # Visualization utilities
│
└── tests/                  # Unit tests


## License

MIT
