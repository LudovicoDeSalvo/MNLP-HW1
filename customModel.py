import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datasets import load_dataset
import wikipedia
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from sklearn.metrics import precision_recall_fscore_support, classification_report
import pickle
import os

# Ensure NLTK dependencies are available
user_input = input("Download NLKT (y/n): ").lower()
if(user_input == 'y'):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# Load dataset
dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset')
label_map = {
    "cultural agnostic": 0,
    "cultural representative": 1,
    "cultural exclusive": 2
}

# Helper: map POS tag to WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Preprocess text into list of tokenized sentences
def preprocess(text):
    if not text:
        return []

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    sentences = sent_tokenize(text.lower())
    processed_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word for word in words if word.isalpha() and word not in stop_words]
        pos_tags = pos_tag(words)
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        if lemmatized:
            processed_sentences.append(lemmatized)

    return processed_sentences

# Get Wikipedia title from Wikidata ID
def getWikipediaPage(wikidataId, lang='en'):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidataId}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        title = data['entities'][wikidataId]['sitelinks'][f'{lang}wiki']['title']
        return title
    except Exception as e:
        print(f"[ERROR] Wikidata fetch failed for {wikidataId}: {e}")
        return None

# Get Wikipedia content from title
def getWikipediaText(title, lang='en'):
    wikipedia.set_lang(lang)
    try:
        return wikipedia.page(title).content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"[WARNING] '{title}' is ambiguous: {e.options[:3]}...")
        return ""
    except wikipedia.exceptions.PageError:
        print(f"[ERROR] Page '{title}' not found.")
        return ""
    except Exception as e:
        print(f"[ERROR] Unexpected error on '{title}': {e}")
        return ""

user_input = input("Download pages from Wikipedia (y/n): ").lower()
if(user_input == 'y'):
    

    # Download and preprocess all items
    items, labels, texts, corpus = [], [], [], []

    counter = 0
    for sample in dataset['train']:
        qid = sample['item'].split('/')[-1]
        label_str = sample['label'].lower()
        label = label_map.get(label_str)
        if label is None:
            continue

        title = getWikipediaPage(qid)
        if not title:
            continue

        text = getWikipediaText(title)
        processed = preprocess(text)
        if not processed:
            continue

        items.append(title)
        labels.append(label)
        texts.append(processed)
        corpus.extend(processed)
        counter=counter+1
        print(f'Counter:{counter}')

    with open('wikiProcessedTexts.pkl', 'wb') as f:
        pickle.dump((items, labels, texts, corpus), f)

with open('wikiProcessedTexts.pkl', 'rb') as f:
    items, labels, texts, corpus = pickle.load(f)



user_input = input("Download pages from Wikipedia for validation (y/n): ").lower()
if(user_input == 'y'):
    # Extend corpus with validation set before training Word2Vec
    val_items, val_labels, val_texts = [], [], []
    for sample in dataset['validation']:
        qid = sample['item'].split('/')[-1]
        label_str = sample['label'].lower()
        label = label_map.get(label_str)
        if label is None:
            continue

        title = getWikipediaPage(qid)
        if not title:
            continue

        text = getWikipediaText(title)
        processed = preprocess(text)
        if not processed:
            continue

        val_items.append(title)
        val_labels.append(label)
        val_texts.append(processed)
        #corpus.extend(processed)  # include validation text in Word2Vec training

    with open('wikiProcessedTextsVal.pkl', 'wb') as f:
        pickle.dump((val_items, val_labels, val_texts, corpus), f)

with open('wikiProcessedTextsVal.pkl', 'rb') as f:
    val_items, val_labels, val_texts, corpus = pickle.load(f)

#print(items,val_items)

# Train Word2Vec model
def train_word2vec(corpus, vector_size=100, window=5, min_count=5):
    return Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4)



# Dataset class
class WikiDataset(Dataset):
    def __init__(self, texts, labels, w2v_model):
        self.labels = labels
        self.vectors = [self.text_to_vec(list(chain.from_iterable(text)), w2v_model) for text in texts]

    def text_to_vec(self, tokens, w2v_model):
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        return torch.tensor(sum(vecs)/len(vecs)) if vecs else torch.zeros(w2v_model.vector_size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]

# Neural classifier
class WikiClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(WikiClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Train function
def train_model(model, dataset, epochs=50, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            X = X.to(device)       # Move inputs to the same device as the model
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Build dataset and train

user_input = input("Train? (y: train / n: load existing): ").lower()
if(user_input == 'y'):
    # Train Word2Vec model
    w2v_model = train_word2vec(corpus)
    w2v_model.save("word2vec.model")

    wiki_dataset = WikiDataset(texts, labels, w2v_model)
    model = WikiClassifier(input_dim=w2v_model.vector_size, hidden_dim=64, num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, wiki_dataset)
    torch.save(model.state_dict(), 'wiki_classifier.pth')
else:
    w2v_model = Word2Vec.load("word2vec.model")

    wiki_dataset = WikiDataset(texts, labels, w2v_model)
    model = WikiClassifier(input_dim=w2v_model.vector_size, hidden_dim=64, num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load('wiki_classifier.pth', map_location=device))

model.to(device)
wiki_val_dataset = WikiDataset(val_texts, val_labels, w2v_model)

model.eval()
model.to(device)

all_preds = []
all_labels = []
with torch.no_grad():
    for X, y in DataLoader(wiki_val_dataset, batch_size=32):
        X = X.to(device)
        y = y.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

print("\nValidation Results:")
print(classification_report(all_labels, all_preds, digits=5))