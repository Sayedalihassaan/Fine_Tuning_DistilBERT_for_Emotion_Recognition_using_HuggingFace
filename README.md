Fine-Tuning DistilBERT for Emotion Recognition

This project demonstrates fine-tuning DistilBERT, a lightweight transformer model, for emotion recognition from text using the HuggingFace Transformers library. The model classifies text into one of six emotions: sadness, anger, love, joy, fear, and surprise. The dataset contains 16,000 labeled text samples, and the fine-tuned model achieves high accuracy for emotion classification.
Table of Contents

Project Overview
Dataset
Features
Installation
Usage
Model Performance
Contributing
License
Acknowledgements

Project Overview
The goal of this project is to fine-tune DistilBERT, a distilled version of BERT, for emotion recognition. By leveraging a labeled dataset of 16,000 text samples, the model learns to classify emotions with high accuracy. The project uses the HuggingFace Transformers library for model training and PyTorch as the backend. Key tasks include:

Data preprocessing and tokenization
Fine-tuning DistilBERT for emotion classification
Evaluating the model using metrics like accuracy, precision, recall, and F1-score
Deploying the model for inference on new text inputs

This project is ideal for applications such as sentiment analysis, chatbots, or mental health monitoring systems.
Dataset
The dataset consists of 16,000 text samples, each labeled with one of six emotions: sadness, anger, love, joy, fear, or surprise. The data is stored in a text file (six emotion.txt) with each line formatted as:
text;emotion

Example:
I feel happy;joy
I am so angry;anger

The dataset is preprocessed to create a pandas DataFrame with two columns: text (input text) and target (emotion label).
Features

Emotion Classification: Classifies text into six emotions using a fine-tuned DistilBERT model.
Efficient Model: Uses DistilBERT for faster inference and lower resource consumption compared to BERT.
Robust Preprocessing: Includes text cleaning, tokenization, and label encoding.
Evaluation Metrics: Provides detailed metrics (accuracy, precision, recall, F1-score) and a confusion matrix.
Inference Pipeline: Includes a detection_text function for predicting emotions on new text inputs.

Installation
To run this project locally, follow these steps:
Prerequisites

Python 3.8+
pip package manager
Google Colab or a local environment with GPU support (optional but recommended)

Steps

Clone the Repository:
git clone https://github.com/your-username/emotion-recognition-distilbert.git
cd emotion-recognition-distilbert


Install Dependencies:Create a virtual environment (optional) and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

The requirements.txt file includes:
torch
transformers
datasets
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
missingno
nltk
wordcloud


Download the Dataset:Place the six emotion.txt dataset in the project directory or update the file path in the notebook to point to your dataset location.

Optional: Mount Google Drive (for Colab users):If using Google Colab, mount your Google Drive to access the dataset:
from google.colab import drive
drive.mount('/content/drive')



Usage

Run the Jupyter Notebook:Open the Fine_Tuning_DistilBERT_for_Emotion_Recognition_using_HuggingFace.ipynb notebook in Jupyter or Google Colab and execute the cells sequentially to:

Load and preprocess the dataset
Fine-tune the DistilBERT model
Evaluate the model
Test the model on sample texts


Test the Model:Use the detection_text function to predict emotions for new text inputs. Example:
sample_text = "I can't stop smiling today, everything feels perfect!"
predicted_emotion = detection_text(sample_text)
print(f"Sentence: {sample_text}\nPredicted class: ( {predicted_emotion} )")

Example output:
Sentence: I can't stop smiling today, everything feels perfect!
Predicted class: ( joy )


Visualize Results:The notebook generates a confusion matrix and classification report to visualize model performance.


Model Performance
The fine-tuned DistilBERT model achieves an accuracy of 93.19% on the test set. Below is the classification report:



Emotion
Precision
Recall
F1-Score
Support



Sadness
0.94
0.92
0.93
435


Anger
0.92
0.88
0.90
418


Joy
0.94
0.95
0.95
1010


Love
0.88
0.85
0.86
306


Fear
0.96
0.97
0.97
943


Surprise
0.71
0.91
0.80
88


Overall Metrics:

Accuracy: 0.9319
Macro Avg: Precision: 0.89, Recall: 0.91, F1-Score: 0.90
Weighted Avg: Precision: 0.93, Recall: 0.93, F1-Score: 0.93

A confusion matrix is also generated to visualize the model's performance across emotion classes.
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please ensure your code follows the project's coding standards and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

HuggingFace: For providing the Transformers library and pre-trained models.
Dataset: The emotion recognition dataset used in this project.
Community: Thanks to the open-source community for valuable resources and tutorials.

