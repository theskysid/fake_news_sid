from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the correct model and tokenizer
model_path = "./models/bert_fakenews"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Input from user
headline = input("📰 Enter a headline to verify: ")

# Tokenize and predict
inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

# Map prediction to label
label_map = {0: "FAKE", 1: "REAL"}
print(f"🧠 Prediction: {label_map[prediction]}")
