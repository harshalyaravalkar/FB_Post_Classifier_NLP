import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

key_path = r'vocal-wavelet-427413-v3-0764ac0731f1.json' 

# Function to load data from Google Sheets
def load_data_from_gsheet(sheet_id, range_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).worksheet(range_name)
    data = get_as_dataframe(sheet, evaluate_formulas=True)
    return data

# Function to save data to Google Sheets
def save_data_to_gsheet(sheet_id, range_name, dataframe):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).worksheet(range_name)
    existing_data = get_as_dataframe(sheet, evaluate_formulas=True)

    # Append new data to existing data
    updated_data = pd.concat([existing_data, dataframe], ignore_index=True)

    # Write updated data back to the sheet
    set_with_dataframe(sheet, updated_data)

# Function to filter rows and update Google Sheets
def filter_and_update_gsheet(sheet_id, range_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).worksheet(range_name)
    data = get_as_dataframe(sheet, evaluate_formulas=True)

    # Filter rows where 'Predicted Label' is 'Lead'
    filtered_data = data[data['Predicted Label'] == 'Lead']

    # Clear existing content and write the filtered data back to the sheet
    sheet.clear()
    set_with_dataframe(sheet, filtered_data)

# Load the trained model and tokenizer
model_save_path = r"trained_model_stuff/model_directory"
tokenizer_save_path = r"trained_model_stuff/tokenizer_directory"
model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_save_path)


# Load the new data from Google Sheets
sheet_id = "1wmcMy_Z7nGcnSVhvLN4TFRbAMaE58H5YO9-nVWcYdEk"  # Replace with your Google Sheet ID
input_range_name = "Sheet4"  # Replace with the sheet name or range name where your data is

new_df = load_data_from_gsheet(sheet_id, input_range_name)
new_df.columns = new_df.columns.str.strip()
new_df = new_df.dropna(subset=['Content'])

# Ensure all text data is valid
new_texts = new_df['Content'].fillna("").astype(str).tolist()

# Prepare the data for prediction
class FBPostsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 8

new_dataset = FBPostsDataset(new_texts, tokenizer, MAX_LEN)
new_data_loader = DataLoader(new_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model to evaluation mode
model.eval()
# Prediction loop
new_predictions = []

with torch.no_grad():
    for batch in new_data_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        new_predictions.extend(preds.cpu().numpy())

# Convert predictions to labels
label_map = {0: 'Lead', 1: 'Other'}
predicted_labels = [label_map[pred] for pred in new_predictions]

# Save predictions to a new Google Sheet
new_df['Predicted Label'] = predicted_labels
output_range_name = "Sheet3"  # Replace with the sheet name or range name where you want to save the predictions
save_data_to_gsheet(sheet_id, output_range_name, new_df)

# Filter rows where 'Predicted Label' is 'Lead' and delete rows where it is 'Other'
filter_and_update_gsheet(sheet_id, output_range_name)

print("Predictions saved to Google Sheet.")
