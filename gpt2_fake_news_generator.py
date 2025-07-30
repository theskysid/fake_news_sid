from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Set pad token to EOS
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Generation loop
while True:
    topic = input("\n📝 Enter a topic for the fake news headline (or type 'exit' to quit): ")
    if topic.lower() == "exit":
        break

    prompt = f"Fake news headline about {topic}:\n"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate headline
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=30,  # 🔧 You can lower this to keep headlines short
            num_return_sequences=1,
            do_sample=True,
            temperature=0.95,
            top_k=60,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    # Decode and trim
    headline = tokenizer.decode(output[0], skip_special_tokens=True)
    headline = headline.split("\n")[0].strip()  # Only first line

    print("\n📰 Fake Headline:\n", headline)
