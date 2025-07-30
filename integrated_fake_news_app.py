from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# -------------------- GPT-2 SETUP --------------------
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.eval()

# Pad token config
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id

# -------------------- BERT CLASSIFIER SETUP --------------------
bert_model_path = "./models/bert_fakenews"
bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path)
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
bert_model.eval()


# -------------------- INFERENCE LOOP --------------------
while True:
    prompt = input("\n📝 Enter a topic (or type 'exit' to quit): ").strip()
    if prompt.lower() == "exit":
        print("👋 Exiting.")
        break

    # --- Generate Fake Headline using GPT-2 ---
    gpt2_inputs = gpt2_tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = gpt2_model.generate(
            gpt2_inputs["input_ids"],
            attention_mask=gpt2_inputs["attention_mask"],
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    headline = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    # --- Predict Using BERT ---
    bert_inputs = bert_tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = bert_model(**bert_inputs).logits
        prediction = torch.argmax(logits, dim=1).item()
        label = "Real" if prediction == 1 else "Fake"

    # --- Print Output ---
    print("\n📰 Generated Headline:\n", headline)
    print("🤖 Prediction:", label)
