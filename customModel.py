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
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

#### Dimensionality settings #####
vector_size = 200
lang_dim = 15
view_dim = 3
total_extra_dim = lang_dim + view_dim
total_dim = vector_size + total_extra_dim

##### Functions #####

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

# Fetch page views
def get_views(title):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title.replace(' ', '_')}/monthly/2023010100/2023123100"
    headers = {
        'User-Agent': 'SapienzaNLP_HW1_Project/1.0 (mailto:your@email.com)'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        monthly = response.json().get("items", [])
        return sum(entry.get("views", 0) for entry in monthly)
    except Exception as e:
        print(f"[FAIL] {title} → {e}")
        return 0

#Fetch number of languages of a page
def get_language_count(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        sitelinks = data['entities'][qid].get('sitelinks', {})
        return len(sitelinks)
    except Exception as e:
        print(f"[ERROR] Failed fetching languages for {qid}: {e}")
        return 0


# Train Word2Vec model
def train_word2vec(corpus, vector_size, window=5, min_count=5):
    return Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4)

#Hyperparameters space searcher
def hyperparameter_tuning():
    
    #setting things up
    hidden_dims = [64, 128, 256]
    learning_rates = [0.001, 0.0005, 0.0002]
    batch_sizes = [32, 64]
    epochs_list = [30, 40, 50, 60, 70, 80, 90]
    # hidden_dims = [128]
    # learning_rates = [0.0005]
    # batch_sizes = [32]
    # epochs_list = [30]

    best_accuracy = 0
    best_model = None
    best_params = {}
    func_len = len(hidden_dims) * len(learning_rates) * len(batch_sizes) * len(epochs_list)
    counter = 1

    for hidden_dim in hidden_dims:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    print(f"\nIteration n.{counter}/{func_len}")
                    print(f"Training with hidden_dim={hidden_dim}, lr={lr}, batch_size={batch_size}, epochs={epochs}")
          
                    # Train loop, executed 5 times
                    acc = 0
                    this_hp_best_model = None

                    for i in range(20):

                        # Create model
                        model = WikiClassifier(input_dim= total_dim , hidden_dim=hidden_dim, num_classes=3)
                        model.to(device)

                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()

                        loader = DataLoader(wiki_dataset, batch_size=batch_size, shuffle=True)

                        for epoch in range(epochs):
                            total_loss = 0
                            for X, y in loader:
                                X = X.to(device)
                                y = y.to(device)
                                optimizer.zero_grad()
                                outputs = model(X)
                                loss = criterion(outputs, y)
                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item()
                            #print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

                        # Evaluate
                        this_hp_acc = evaluate_model(model)
                        if(this_hp_acc > acc):
                            acc = this_hp_acc
                            this_hp_best_model = model                            
                            
                    #saving best overall
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = this_hp_best_model.state_dict()
                        best_params = {
                            'hidden_dim': hidden_dim,
                            'lr': lr,
                            'batch_size': batch_size,
                            'epochs': epochs,
                        }

                    counter = counter + 1
                        
    return best_model,best_params

#Evaluator of a model
def evaluate_model(model):
    model.eval()
    all_preds = []
    all_labels_eval = []
    with torch.no_grad():
        for X_val, y_val in DataLoader(wiki_val_dataset, batch_size=32):
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs = model(X_val)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels_eval.extend(y_val.tolist())

    # Calculate accuracy
    correct = sum([p == l for p, l in zip(all_preds, all_labels_eval)])
    acc = correct / len(all_labels_eval)
    
    return acc

# Train function
def train_model(model, dataset, epochs=50, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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





##### Classes #####

# Dataset class
class WikiDataset(Dataset):
    def __init__(self, texts, labels, views, langs, w2v_model):
        self.labels = labels
        normalized_views = (np.array(views) - np.mean(views)) / (np.std(views) + 1e-6)
        normalized_langs = (np.array(langs) - np.mean(langs)) / (np.std(langs) + 1e-6)
        self.views_tensor = torch.tensor(normalized_views, dtype=torch.float32).unsqueeze(1).repeat(1, view_dim)
        self.langs_tensor = torch.tensor(normalized_langs, dtype=torch.float32).unsqueeze(1).repeat(1, lang_dim)
        self.vectors = [self.text_to_vec(list(chain.from_iterable(text)), w2v_model) for text in texts]

    def text_to_vec(self, tokens, w2v_model):
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        return torch.tensor(sum(vecs)/len(vecs)) if vecs else torch.zeros(w2v_model.vector_size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        combined = torch.cat([self.vectors[idx], self.views_tensor[idx], self.langs_tensor[idx]], dim=0)
        return combined, self.labels[idx]

# Neural classifier
class WikiClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(WikiClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.output(x)



##### Main #####

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

#Download Wikipedia Pages
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


#Download Wikipedia Pages of validation
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


# Fetch or load training views
if os.path.exists("train_views.pkl"):
    with open("train_views.pkl", "rb") as f:
        train_views = pickle.load(f)
    print("[TRAIN] Views loaded from file.")
else:
    train_views = []
    print("\n[TRAIN] Fetching page views...")
    for i, title in enumerate(items):
        views = get_views(title)
        print(f"[{i+1}/{len(items)}] {title} → {views}")
        train_views.append(views)
    with open("train_views.pkl", "wb") as f:
        pickle.dump(train_views, f)

# Fetch or load validation views
if os.path.exists("val_views.pkl"):
    with open("val_views.pkl", "rb") as f:
        val_views = pickle.load(f)
    print("[VAL] Views loaded from file.")
else:
    val_views = []
    print("\n[VAL] Fetching page views...")
    for i, title in enumerate(val_items):
        views = get_views(title)
        print(f"[{i+1}/{len(val_items)}] {title} → {views}")
        val_views.append(views)
    with open("val_views.pkl", "wb") as f:
        pickle.dump(val_views, f)


if os.path.exists("train_langs.pkl"):
    with open("train_langs.pkl", "rb") as f:
        train_langs = pickle.load(f)
    print("[TRAIN] Languages loaded from file.")
else:
    train_langs = []
    print("\n[TRAIN] Fetching Wikipedia languages...")
    for i, sample in enumerate(dataset['train']):
        qid = sample['item'].split('/')[-1]
        lang_count = get_language_count(qid)
        print(f"[{i+1}/{len(dataset['train'])}] {qid} → {lang_count} languages")
        train_langs.append(lang_count)
    with open("train_langs.pkl", "wb") as f:
        pickle.dump(train_langs, f)

if os.path.exists("val_langs.pkl"):
    with open("val_langs.pkl", "rb") as f:
        val_langs = pickle.load(f)
    print("[VAL] Languages loaded from file.")
else:
    val_langs = []
    print("\n[VAL] Fetching Wikipedia languages...")
    for i, sample in enumerate(dataset['validation']):
        qid = sample['item'].split('/')[-1]
        lang_count = get_language_count(qid)
        print(f"[{i+1}/{len(dataset['validation'])}] {qid} → {lang_count} languages")
        val_langs.append(lang_count)
    with open("val_langs.pkl", "wb") as f:
        pickle.dump(val_langs, f)



# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
finalModel = None

user_input = input("Train? (y: train / n: load existing): ").lower()
if user_input == 'y':

     #Train word2vec
    w2v_model = train_word2vec(corpus,vector_size)
    
    #build dataset
    wiki_dataset = WikiDataset(texts, labels, train_views, train_langs, w2v_model)
    wiki_val_dataset = WikiDataset(val_texts, val_labels, val_views, val_langs, w2v_model)

    #Hyperparameter search
    best_model, best_params = hyperparameter_tuning()

    finalModel = WikiClassifier(input_dim= total_dim, hidden_dim=best_params['hidden_dim'], num_classes=3)
    finalModel.load_state_dict(best_model)
    finalModel.to(device)

    w2v_model.save("word2vec.model")
    torch.save(best_model, 'wiki_classifier.pth')
    
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

     
else:
    w2v_model = Word2Vec.load("word2vec.model")
    with open("best_params.json", "r") as f:
        best_params = json.load(f)

    finalModel = WikiClassifier(input_dim= vector_size + total_extra_dim, hidden_dim=best_params['hidden_dim'], num_classes=3)
    finalModel.load_state_dict(torch.load('wiki_classifier.pth'))
    finalModel.to(device)

    wiki_val_dataset = WikiDataset(val_texts, val_labels, val_views, val_langs, w2v_model)


##### EVALUATION #####

finalModel.eval()
all_preds = []
all_labels_eval = []
with torch.no_grad():
    for X_val, y_val in DataLoader(wiki_val_dataset, batch_size=32):
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        outputs = finalModel(X_val)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels_eval.extend(y_val.tolist())

print("\nReport:")
print(classification_report(all_labels_eval, all_preds, digits=5))

# Confusion matrix
label_map = ["cultural agnostic", "cultural representative", "cultural exclusive"]
cm = confusion_matrix(all_labels_eval, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map, yticklabels=label_map)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()