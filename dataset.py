import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class QueryDataset(Dataset):
    def __init__(self, data_path, split, bert_type, max_length=512):
        self.data = pd.read_csv(os.path.join(data_path, f'{split}.csv'))
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx]['query']
        label = self.data.iloc[idx]['label']
        
        # Tokenize the query
        encoding = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(data_path, split, bert_type, batch_size, max_length):
    dataset = QueryDataset(data_path, split, bert_type, max_length)
    if split == 'train':
        return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    

def generate_data_loader(data_path, batch_size, max_length, bert_type='huawei-noah/TinyBERT_General_4L_312D'):
    train_loader = create_data_loader(data_path, 'train', bert_type, batch_size, max_length)
    val_loader = create_data_loader(data_path, 'val', bert_type, batch_size, max_length)
    test_loader = create_data_loader(data_path, 'test', bert_type, batch_size, max_length)
    return train_loader, val_loader, test_loader
