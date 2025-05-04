# MNLP Homework 1 — Cultural Concept Classification

This repository contains the implementation of two classification pipelines developed for the **MNLP 2025 Homework 1**.  
The goal is to classify Wikidata entities into three cultural categories:

- **Cultural Agnostic**
- **Cultural Representative**
- **Cultural Exclusive**

---

## Contents

###  **Language Model (LM) Approach**

- **`roBERTa-AutoTrainer.py`**  
  Final version of the LM-based classifier using HuggingFace's Trainer API.  
  Includes optional hyperparameter tuning via Optuna.

- **`roBERTa.py`**  
  Initial version of the LM model.  
  This is a deprecated implementation without tuning functionality.

The LM pipeline is based on **roberta-base** and fine-tunes the model using the following metadata fields from each item:

- Name
- Description
- Type
- Category
- Subcategory

All fields are concatenated with `[SEP]` tokens and fed to the transformer model for classification.

---

###  **Non-LM Approach (Word2Vec + Page Views)**

- **`customModel.py`**  
  Final implementation of the non-LM model.  
  This hybrid approach combines semantic representations from Wikipedia text (via Word2Vec) with normalized page view statistics from Wikimedia.

The script:

- Preprocesses Wikipedia content using tokenization, POS tagging, stopword removal, and lemmatization.
- Trains multiple Word2Vec embeddings over varying dimensions.
- Concatenates embedding vectors with repeated view-based features.
- Trains a feedforward neural network with hyperparameter tuning (grid search).

---

## Output

- Trained models are stored in `.pth` format.
- Word2Vec embeddings and all intermediate preprocessed data are saved in `.pkl` format.
- Classification results and confusion matrices are available in the appendix of the LaTeX report.
