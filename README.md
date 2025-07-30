# fake_news_sid

# 📰 Fake News Detection using GPT-2 and BERT

This project explores how AI can both generate and detect fake news. It uses **GPT-2** to create synthetic (always fake) news headlines and **BERT** to classify them as real or fake. The goal is to highlight the ethical implications of generative AI and showcase its responsible use in combating misinformation.

> ⚠️ **Note:** This project always generates fake news using GPT-2.  
> 📁 **Model and data folders are not included** due to size constraints.
> It is trained using the dataset provided with link and run the combined.py on the dataset

---

## 📁 Project Structure

Fake-News-Detection/
├── generation/ # GPT-2 headline generation scripts
├── detection/ # BERT classification scripts
├── requirements.txt # Python dependencies
├── README.md # Project documentation



---

## 📊 Dataset

The model is trained using the following dataset:

📥 **Download Link:** https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Extract the dataset into a `data/` folder inside the project directory.

---

## 🔧 Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/theskysid/Fake-News-Detection.git
cd Fake-News-Detection
