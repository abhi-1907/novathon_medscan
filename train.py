import torch
from transformers import BertTokenizer, BertForTokenClassification

# Function to load the trained model
def load_model(model_path):
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Function to extract entities from the text
def extract_entities(text, model, tokenizer):
    # Tokenize the input text
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Predict labels for each token
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Decode tokens and get their corresponding labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = predictions[0].cpu().numpy()
    
    # Map tokens to their labels
    entity_dict = {}
    for token, label in zip(tokens, labels):
        if label != 0:  # Assuming 0 is for non-entity tokens
            label_name = model.config.id2label[label]
            if label_name not in entity_dict:
                entity_dict[label_name] = []
            entity_dict[label_name].append(token)
    
    return entity_dict

# Example usage
if __name__ == "__main__":
    # Load the trained model and tokenizer
    model, tokenizer = load_model('D:\\novathonnew\\trained_bert_model')
    
    # Input paragraph (random words)
    text = "John Doe, a 45 years old male with Hypertension, was admitted to Mercy Hospital on January 10th. He was diagnosed with high blood pressure and prescribed medication."
    
    # Extract entities
    extracted_entities = extract_entities(text, model, tokenizer)
    
    # Print the extracted entities
    print("Extracted Entities:")
    for entity, values in extracted_entities.items():
        print(f"{entity}: {', '.join(values)}")
