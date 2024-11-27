---
title: "Streamlit Food Product Classifier"
emoji: "🍏"

sdk: "streamlit"
sdk_version: "1.21.0"
app_file: "app.py"
pinned: false
---

Key Updates:
The SDK Version section specifies the required version (1.20.0 or 1.21.0).
The Dependencies section includes all the necessary libraries for the project.
Basic instructions on how to install dependencies with poetry or pip, and run the Streamlit app.

# Streamlit Food Product Classifier

This is a simple Streamlit-based web application that uses machine learning models for food product classification. It allows users to upload an image of a food product, which is then classified into one of several categories using a deep learning model.

## Features
- **Food Image Classification**: Upload food images to classify them into predefined categories.
- **Model**: The model used is a custom CNN trained on food product images, utilizing a `ResNet50` backbone for better accuracy.
- **Prediction with Confidence Scores**: After uploading an image, the model predicts the food category and provides the prediction's confidence score.

## Technologies Used
- **Streamlit**: For building the interactive web application.
- **fastai**: For model training and inference, leveraging the `cnn_learner` API.
- **torch**: For deep learning and model architecture.
- **ResNet50**: A pre-trained model used as a backbone for transfer learning.
- **Pillow**: For handling image processing.

## Installation

To run the project locally, you can follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/food-product-classifier.git
   cd food-product-classifier
