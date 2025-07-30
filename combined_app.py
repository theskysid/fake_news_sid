from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load DistilBERT model and tokenizer
bert_model_path = "./models/bert_fakenews"
bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path)
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)

# Function to generate fake news headline
def generate_fake_news(prompt, max_length=50):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

# Function to detect if text is fake or real
def detect_fake_news(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Fake" if prediction == 1 else "Real"

# Example usage
if __name__ == "__main__":
    prompt = "Breaking news:"
    generated_headline = generate_fake_news(prompt)
    print("Generated Headline:", generated_headline)

    prediction = detect_fake_news(generated_headline)
    print("Prediction:", prediction)
