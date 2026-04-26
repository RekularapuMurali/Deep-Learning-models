import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from data.medical_data import MEDICAL_TEXTS, LABEL2IDX, NUM_LABELS

MODEL_NAME = 'bert-base-uncased'
MAX_LEN    = 64
BATCH_SIZE = 8
EPOCHS     = 5
SAVE_DIR   = 'model'

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, max_length=MAX_LEN,
                                   padding='max_length', truncation=True,
                                   return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

# Prepare data
texts  = [t for t, _ in MEDICAL_TEXTS] * 8
labels = [LABEL2IDX[l] for _, l in MEDICAL_TEXTS] * 8

combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

print(f"Total samples: {len(texts)}")
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

dataset  = MedicalDataset(list(texts), list(labels), tokenizer)
n_val    = int(len(dataset) * 0.1)
n_train  = len(dataset) - n_val
train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

print("Loading BERT model...")
model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch   = batch['labels'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels_batch).sum().item()
        total   += len(labels_batch)

    train_acc = correct / total * 100

    # ── Validate ──
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch   = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == labels_batch).sum().item()
            val_total   += len(labels_batch)

    val_acc = val_correct / val_total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Model saved!")