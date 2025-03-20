#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating embeddings for text and images.
"""

import pandas as pd
import numpy as np
from typing import List

def generate_embeddings(df: pd.DataFrame, 
                        text_model: str = "bert-base-uncased",
                        image_model: str = "resnet50") -> pd.DataFrame:
    """
    Generate embeddings for text and images in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the dataset
        text_model (str, optional): Name of the text embedding model
        image_model (str, optional): Name of the image embedding model
    
    Returns:
        pd.DataFrame: DataFrame with added embedding columns
    """
    # Create a copy to avoid modifying the original
    results_df = df.copy()
    
    # Generate text embeddings if they don't exist
    if 'text_embeddings' not in results_df.columns:
        print(f"Generating text embeddings using {text_model}...")
        results_df['text_embeddings'] = generate_text_embeddings(
            results_df['structured_caption'].tolist(), model_name=text_model
        )
    
    # Generate image embeddings if they don't exist
    if 'Image_Embedding' not in results_df.columns:
        print(f"Generating image embeddings using {image_model}...")
        results_df['Image_Embedding'] = generate_image_embeddings(
            results_df['imgs'].tolist(), model_name=image_model
        )
    
    return results_df

def generate_text_embeddings(texts: List[str], model_name: str = "bert-base-uncased") -> List[np.ndarray]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts (List[str]): List of texts to embed
        model_name (str, optional): Name of the model to use
    
    Returns:
        List[np.ndarray]: List of embedding arrays
    """
    try:
        # Import the necessary libraries
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Generate embeddings
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                    max_length=512, return_tensors='pt')
            
            with torch.no_grad():
                model_output = model(**encoded_input)
                
            # Use the CLS token as the sentence embedding
            batch_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    except ImportError:
        print("Warning: transformers package not found. Using random embeddings.")
        # Generate random embeddings for demonstration
        return [np.random.rand(768) for _ in texts]

def generate_image_embeddings(image_paths: List[str], model_name: str = "resnet50") -> List[np.ndarray]:
    """
    Generate embeddings for a list of images.
    
    Args:
        image_paths (List[str]): List of paths to images
        model_name (str, optional): Name of the model to use
    
    Returns:
        List[np.ndarray]: List of embedding arrays
    """
    try:
        # Import the necessary libraries
        import torch
        from torchvision import models, transforms
        from PIL import Image
        
        # Load the model
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            # Remove the classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Define image preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        
        # Generate embeddings
        embeddings = []
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                
                with torch.no_grad():
                    embedding = model(img_tensor).squeeze().numpy()
                
                embeddings.append(embedding.flatten())
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Use a zero vector as a placeholder
                embeddings.append(np.zeros(2048))
        
        return embeddings
    
    except ImportError:
        print("Warning: PyTorch or torchvision not found. Using random embeddings.")
        # Generate random embeddings for demonstration
        return [np.random.rand(2048) for _ in image_paths]
