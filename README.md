# Facebook Post Classifier for Lead Generation

## Overview
This project automates the classification of scraped Facebook posts into two categories: **Leads** and **Other Content**. The tool integrates with Google Sheets to fetch post data, processes the text using a machine learning model, and appends the filtered leads back into another Google Sheet. This solution is ideal for businesses or teams looking to automate the lead generation process from large datasets of Facebook posts.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)

## Features
- **Data Integration**: Fetches scraped Facebook post data from Google Sheets.
- **Text Classification**: Utilizes pre-trained Transformer models (DistilBERT) to classify posts.
- **Automated Workflow**: Appends identified leads to a separate Google Sheet for easy access.
- **Customizable**: Can be adapted for different classification tasks or social media data.

## Tech Stack
- **Google Sheets API**: For interacting with Google Sheets to fetch and update data.
- **Transformers (Hugging Face)**: Pre-trained DistilBERT for text classification.
- **Torch**: Deep learning framework for implementing the model.
- **gspread**: Python library for Google Sheets API integration.

## Installation
To get started with the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/facebook-post-classifier.git
   cd facebook-post-classifier
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Set up your Google API credentials and save them in a credentials.json file in the root directory.

Note: I haven't put the API credentials I used as they get disabled when posted on public repositories.

## Model File
Due to file size limitations, the trained model is hosted on Google Drive. You can download it [here](https://drive.google.com/drive/folders/1I2ULcdcOv2-7HmrtC4vJ4USKtVUDubx7?usp=drive_link).

Click on the title "trained_model_stuff" and download the folder. After downloading, place the folder in the same directory as the py file.


## Usage
Once the setup is complete, you can run the script to classify Facebook posts:
    
    python classify_posts.py

The script will:

1. Fetch Facebook post data from a specified Google Sheet.
2. Classify posts into Leads and Other Content using a pre-trained DistilBERT model.
3. Append the filtered leads into another Google Sheet.

## Configuration
You can modify the configuration for the Google Sheets integration and classification parameters in the script. Ensure that you specify the correct:

- Google Sheets IDs and ranges for input and output data.
- Paths to your "credentials.json" for Google API authentication.

## How It Works

1. **Google Sheets Integration**: The script uses the Google Sheets API to pull data from a Google Sheet containing Facebook posts.
2. **Classification**: Using a Transformer model (DistilBERT), the text of each post is classified as either a Lead or Other Content.
3. **Appending Results**: The identified leads are appended to another Google Sheet for easy review and use.
