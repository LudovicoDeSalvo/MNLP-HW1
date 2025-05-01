# Imports
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score


# Load dataset
dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset')

#Setting up base elements
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
model_path = "./roBERTa_model_AUTO"
label_map = {
    "cultural agnostic": 0,
    "cultural representative": 1,
    "cultural exclusive": 2
}

# Custom Dataset Class
class CulturalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['name'] + " [SEP] " + sample['description'] + " [SEP] " + sample['type'] + " [SEP] " + sample['category'] + " [SEP] " + sample['subcategory']
        label = label_map[sample['label'].lower()]

        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length)
        inputs['labels'] = label
        return inputs

# Prepare train and validation datasets
train_dataset = CulturalDataset(dataset['train'], tokenizer)
eval_dataset = CulturalDataset(dataset['validation'], tokenizer)

# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# Model init for fresh instance each trial
def model_init():
    return RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Hyperparameter search space
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 7),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }

# Objective function for optimization
def compute_objective(metrics):
    return metrics["eval_f1"]

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_path,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    model_init=model_init,
)

# Tunining
hp_tuned = False

user_input = input("Do hyperparameter tuning? y/n: ").lower()
if user_input == 'y':

    #setting up hp search parameters
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        n_trials=60,
        hp_space=hp_space,
        compute_objective=compute_objective,
    )
    
    print("\nBest hyperparameters found:")
    print(best_trial)

    # Setting found hyperparameters
    best_args = training_args
    best_args.learning_rate = best_trial.hyperparameters["learning_rate"]
    best_args.per_device_train_batch_size = best_trial.hyperparameters["per_device_train_batch_size"]
    best_args.num_train_epochs = best_trial.hyperparameters["num_train_epochs"]
    best_args.weight_decay = best_trial.hyperparameters["weight_decay"]

    trainer.args = best_args
    hp_tuned = True

#Training
if hp_tuned == True :
    user_input = 'y'
else:
    user_input = input("Train (DO if not done before)? y/n: ").lower()

if (user_input == 'y') :   
    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
else:
    #load existing model
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

##### EVALUATION #####

predictions = trainer.predict(eval_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print("\nReport:")
print(classification_report(labels, preds, digits=5))