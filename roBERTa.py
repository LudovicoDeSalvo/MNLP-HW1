#Imports
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report

#Loading Dataset and defining base model (roberta), tokenizer and labels
dataset = load_dataset('sapienzanlp/nlp2025_hw1_cultural_dataset')

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
labels = [0,1,2] #Agnostic, Representive, Exclusive


#print(dataset['train'][332]) 
# {
#    'item': 'http://www.wikidata.org/entity/Q252187',
#    'name': 'áo dài',
#    'description': 'Vietnamese national costume, tunic',
#    'type': 'concept',
#    'category': 'fashion',
#    'subcategory': 'clothing',
#    'label': 'cultural representative'
# }

#Creating custom dataset class
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
        label = sample['label']
        label_map = {
            "cultural agnostic": 0,
            "cultural representative": 1,
            "cultural exclusive": 2
        }
        label_id = label_map[label.lower()]  

        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(label_id)
        return item

#defining dataloader
customDataset = CulturalDataset(dataset['train'], tokenizer)
dataloader = DataLoader(customDataset, batch_size=8, shuffle=True)

#optimizing
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = len(dataloader) * 4  # 4 epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#using gpu
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Do training?")
user_input = input("Y/n: ")

if(user_input == 'y'):
#training

    model.train()
    epochs = 4

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

    #saving model
    model.save_pretrained("./cultural_roberta_model")
    tokenizer.save_pretrained("./cultural_roberta_model")


#loading trained set

model_path = "./cultural_roberta_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

model.eval()
model.to(device)

eval_dataset = CulturalDataset(dataset['validation'], tokenizer)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)


#evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

all_preds = torch.tensor(all_preds, dtype=torch.long)
all_labels = torch.tensor(all_labels, dtype=torch.long)

# # Accuracy
# accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
# print(f"Accuracy: {accuracy:.4f}")

# # Precision, Recall, F1
# precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")

# Optional: Pretty table
print("\nDetailed classification report:")
print(classification_report(all_labels, all_preds))

