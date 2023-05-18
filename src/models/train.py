from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_modules import CustomDataset, TextClassifier
from transformers import BertModel, BertTokenizer
import sys
sys.append("../")
from data.data_preprocessing import data_preprocessing



#hyper parameters
lr = 2e-5
batch_size = 4

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#load data
PATH = "../../data/raw.txt"
train_data, val_data = data_preprocessing(PATH)
train_texts = train_data.text
train_labels = train_data.label
val_texts = val_data.text
val_labels = val_data.label

train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=128)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model = TextClassifier(model, learning_rate=lr)

trainer = pl.Trainer(gpus=1, max_epochs=10)
trainer.fit(model, train_dataloader, val_dataloader)