from transformers import BertForTokenClassification, BertTokenizerFast
import torch

# Load fine-tuned model and tokenizer
model = BertForTokenClassification.from_pretrained("./ner_model")
tokenizer = BertTokenizerFast.from_pretrained("./ner_model")

# Create a mapping from label indices to label names
label_map = {0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

# Define the evaluate_model function
def evaluate_model(paragraph):
    # Tokenize the input paragraph
    encoding = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Get model predictions
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Get predicted labels
    predictions = torch.argmax(logits, dim=-1)
    
    # Convert input_ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Print the token, predicted label, and actual label
    for token, pred in zip(tokens, predictions[0].cpu().numpy()):
        print(f"Token: {token}, Predicted: {label_map.get(pred, 'Unknown')}")

# Example test
evaluate_model("John is 45 years old and is being treated for diabetes at General Hospital.")
