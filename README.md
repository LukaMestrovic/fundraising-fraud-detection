# FRAUD BUSTER

## Introduction

FRAUD BUSTER is an AI-powered fraudster detector designed to tackle the problem of fraudulent campaigns in online fundraising. This repository contains the code and resources for training a model to classify campaigns as "fraud" or "not-fraud" based on their textual content.

## Dataset

We have created a custom dataset containing campaign data. The dataset is located at **data/raw.txt**. It consists of approximately 90 "not-fraud" campaigns and 20 "fraud" campaigns. Not fraud campaigns were extracted from GoFundMe website and fraud campaigns were created by ChatGPT. Each campaign data entry includes the following fields: "mail", "text", and "label".

## Model Training

The model was trained using a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. The data preprocessing steps can be found in **src/data/data_preprocessing.py**. The model training script is located at **src/models/train.py**. 

The best model achieved during training was saved in the **models/** directory.

## Usage

To use FRAUD BUSTER, follow these steps:

1. Install the necessary dependencies mentioned in the `environment.yml` file.
2. Preprocess the data using the script **src/data/data_preprocessing.py**.
3. Train the model using the script **src/models/train.py**.
4. Once trained, the best model will be saved in the **models/** directory.
5. Use the trained model to classify new campaign data.

## Future Work

We are continuously working on improving FRAUD BUSTER. Some areas for future enhancements include:

- Increasing the dataset size to improve model performance.
- Exploring different architectures and fine-tuning techniques.
- Incorporating additional features beyond the campaign text for improved fraud detection.

