import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
data_path = r"D:\novathonnew\records\healthcare_dataset\healthcare_dataset.csv"  # Update with your dataset path
data = pd.read_csv(data_path)

# Check the unique values in the 'Test Results' column
print("Unique Test Results:", data['Test Results'].unique())

# Token classification labels mapping
label_mapping = {
    "Normal": 0,
    "Abnormal": 1,
    "Inconclusive": 2,
    # Add any other unique test results if necessary
}
num_labels = len(label_mapping)


class HealthcareDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, label_mapping):
        self.data = data
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["Medical Condition"]  # Or another column that represents the text
        label = self.label_mapping[row["Test Results"]]  # Map the 'Test Results' to a label

        # Tokenize the text
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}

        # Create labels tensor
        labels = torch.full((inputs["input_ids"].shape[0],), -100, dtype=torch.long)  # Initialize with -100 (ignore index)
        
        # Set the label for each token (here you can apply the logic of assigning the label to all tokens, or the first one, etc.)
        # If you want the label to be applied to all tokens:
        labels[:] = label

        item["labels"] = labels

        return item

# Load the tokenizer and model
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels).to(device)


# Split data into train and test
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Create Dataset objects
train_dataset = HealthcareDataset(train_data, tokenizer, label_mapping)
val_dataset = HealthcareDataset(val_data, tokenizer, label_mapping)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Match the batch size in DataLoader
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none",  # Disable reporting to WandB
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()


# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Model saved successfully.")
