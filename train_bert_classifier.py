import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments


# ✅ Load the data
df = pd.read_csv('Data/combined_news.csv')
df = df[['text', 'label']]  # use the cleaned 'text' column

print(df['label'].value_counts())
# ✅ Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# ✅ Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# ✅ Tokenize the text
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# ✅ Dataset Class
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# ✅ Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./models/bert_fakenews',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=50,
)


# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ✅ Train the model
trainer.train()

# Evaluate on validation set
predictions = trainer.predict(val_dataset)
pred_labels = torch.argmax(torch.tensor(predictions.predictions), axis=1)
true_labels = torch.tensor(val_labels)

print("\n📊 Validation Accuracy:", accuracy_score(true_labels, pred_labels))
print("\n📄 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=["True", "Fake"]))

# ✅ Save the model and tokenizer
model.save_pretrained('./models/bert_fakenews')
tokenizer.save_pretrained('./models/bert_fakenews')

print("✅ Model training complete and saved to ./models/bert_fakenews")
