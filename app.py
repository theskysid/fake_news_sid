from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Use pretrained tokenizer (not your local path)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./models/bert_fakenews")  # Local fine-tuned model

def predict_fake_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    return "Fake" if prediction == 1 else "Real"

# Test
text = "NASA announces discovery of water on Mars."
print("✅ Predicted:", predict_fake_news(text))
