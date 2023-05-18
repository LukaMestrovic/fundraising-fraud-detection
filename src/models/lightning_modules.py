import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class TextClassifier(pl.LightningModule):
    def __init__(self, bert_model, learning_rate=2e-5):
        super(TextClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self.forward(input_ids, attention_mask)
        outputs = outputs.flatten()
        outputs = outputs.float()
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        self.train_acc.update(outputs, labels)

        self.log('train_loss', loss)
        return loss
    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        self.log("train_accuracy", train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self.forward(input_ids, attention_mask)
        outputs = outputs.flatten()
        outputs = outputs.float()
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        self.val_acc.update(outputs, labels)

        self.log('val_loss', loss, prog_bar=True)
    
    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        self.log("val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer