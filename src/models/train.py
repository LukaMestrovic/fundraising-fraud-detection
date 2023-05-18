from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_modules import CustomDataset, TextClassifier
from transformers import BertModel, BertTokenizer
import os
import sys
sys.path.append("../")
from data.data_preprocessing import data_preprocessing



#hyper parameters
lr = 2e-5
batch_size = 1
epochs = 10
num_workers = os.cpu_count()
max_length = 32

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#load data
PATH = "../../data/raw.txt"
train_data, val_data = data_preprocessing(PATH)
train_texts = train_data['text']
train_labels = train_data['label']
val_texts = val_data['text']
val_labels = val_data['label']

train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=max_length)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

model = TextClassifier(model, learning_rate=lr)

trainer = pl.Trainer(max_epochs=epochs, precision=16, accelerator='gpu')
trainer.fit(model, train_dataloader, val_dataloader)