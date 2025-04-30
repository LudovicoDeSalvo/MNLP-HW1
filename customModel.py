import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datasets import load_dataset
import requests
import wikipedia
import re
import nltk
nltk.download('punkt')  # Downloads tokenizer models
from nltk.tokenize import word_tokenize

Simulated dataset: list of dicts with QID, name, category
Normally you'd extract this from the cultural dataset
sample_data = [
  {"qid": "Q252187", "name": "áo dài", "category": "clothing", "wiki_title": "Áo_dài"},
    {"qid": "Q11299", "name": "t-shirt", "category": "clothing", "wiki_title": "T-shirt"},
    {"qid": "Q213434", "name": "kente cloth", "category": "clothing", "wiki_title": "Kente_cloth"},
    {"qid": "Q15772", "name": "sushi", "category": "food", "wiki_title": "Sushi"},
    {"qid": "Q27436", "name": "pizza", "category": "food", "wiki_title": "Pizza"},
    {"qid": "Q779", "name": "injera", "category": "food", "wiki_title": "Injera"},
]

dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset')

# Step 1: Get Wikipedia pageviews
def get_pageviews(title, lang="en"):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{title}/monthly/20230101/20231231"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        total_views = sum(item['views'] for item in data['items'])
        return total_views
    else:
        return None

# Step 2: Attach views to each item
for item in sample_data:
    item["views"] = get_pageviews(item["wiki_title"])

# Step 3: Compute category-wise stats
category_visits = defaultdict(list)
for item in sample_data:
    if item["views"] is not None:
        category_visits[item["category"]].append(item["views"])

category_stats = {
    category: (np.mean(views), np.std(views)) for category, views in category_visits.items()
}

# Step 4: Classify each item based on dynamic thresholds
for item in sample_data:
    views = item.get("views")
    category = item["category"]
    if views is None:
        item["label_pred"] = "unknown"
        continue

    mean, std = category_stats[category]
    if views >= mean + std:
        item["label_pred"] = "cultural agnostic"
    elif views >= mean - std:
        item["label_pred"] = "cultural representative"
    else:
        item["label_pred"] = "cultural exclusive"

        def get_pageviews(title, lang="en"):
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{title}/monthly/20230101/20231231"
    print(f"Calling: {url}")  # DEBUG
    r = requests.get(url)
    print(f"Status: {r.status_code}")  # DEBUG
    if r.status_code == 200:
        data = r.json()
        total_views = sum(item['views'] for item in data['items'])
        return total_views
    else:
        print("FAILED:", r.text)
        return None


# Display results
df = pd.DataFrame(sample_data)
print(df[["name", "category", "views", "label_pred"]])




#########################################################
#########################################################
#########################################################





def getWikipediaPage(wikidataId, lang='en'):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidataId}.json"
    response = requests.get(url).json()
    try:
        title = response['entities'][wikidataId]['sitelinks'][f'{lang}wiki']['title']
        return title
    except KeyError:
        return None

def getWikipediaText(title, lang='en'):
    wikipedia.set_lang(lang)
    try:
        return wikipedia.page(title).content
    except:
        return ""
    
def preprocess(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if len(t) > 2]

from gensim.models import Word2Vec

def train_word2vec(corpus, vector_size=100, window=5, min_count=5):
    return Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4)

import torch
from torch.utils.data import Dataset

#dataset for classification
class WikiDataset(Dataset):
    def __init__(self, texts, labels, w2v_model):
        self.labels = labels
        self.vectors = [self.text_to_vec(text, w2v_model) for text in texts]

    def text_to_vec(self, tokens, w2v_model):
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        return torch.tensor(sum(vecs)/len(vecs)) if vecs else torch.zeros(w2v_model.vector_size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]
    
#classifier
import torch.nn as nn

class WikiClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(WikiClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

#training
from torch.utils.data import DataLoader
import torch.optim as optim

def train_model(model, dataset, epochs=10, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#save model
torch.save(model.state_dict(), 'classifier.pth')