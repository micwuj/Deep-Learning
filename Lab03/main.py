import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import f1_score
import copy
from tqdm import tqdm
import numpy as np

class DataPreprocessor:
    def __init__(self, train_path='en-ner-conll-2003/train/train.tsv',
                 dev_x_path='en-ner-conll-2003/dev-0/in.tsv',
                 dev_y_path='en-ner-conll-2003/dev-0/expected.tsv'):
        
        print("Loading data...")
        self.train = pd.read_csv(train_path, sep='\t', names=['label', 'text'])
        self.dev_x = pd.read_csv(dev_x_path, sep='\t', names=['text'])
        self.dev_y = pd.read_csv(dev_y_path, sep='\t', names=['label'])
        
        # Tokenize
        self.train['tokens'] = self.train['text'].apply(lambda x: x.split())
        self.train['labels'] = self.train['label'].apply(lambda x: x.split())
        self.dev_x['tokens'] = self.dev_x['text'].apply(lambda x: x.split())
        self.dev_y['labels'] = self.dev_y['label'].apply(lambda x: x.split())
        
        self._verify_data()
        
        # Build vocabularies
        self.word_to_ix = self._build_word_to_ix(self.train['tokens'])
        self.tag_to_ix = self._build_tag_to_ix(self.train['labels'])
        
        print(f"Vocabulary size: {len(self.word_to_ix)}")
        print(f"Tag set size: {len(self.tag_to_ix)}")
        print(f"Training samples: {len(self.train)}")
        print(f"Dev samples: {len(self.dev_x)}")
    
    def _verify_data(self):
        mismatches = 0
        for text, labels in zip(self.train['tokens'], self.train['labels']):
            if len(text) != len(labels):
                mismatches += 1
        
        if mismatches > 0:
            print(f"Warning: Found {mismatches} text-label length mismatches in training data")
    
    def _build_word_to_ix(self, tokens_list):
        word_to_ix = {"<PAD>": 0, "<UNK>": 1}
        for tokens in tokens_list:
            for token in tokens:
                if token not in word_to_ix:
                    word_to_ix[token] = len(word_to_ix)
        return word_to_ix
    
    def _build_tag_to_ix(self, labels_list):
        tag_to_ix = {"<PAD>": 0}
        for labels in labels_list:
            for label in labels:
                if label not in tag_to_ix:
                    tag_to_ix[label] = len(tag_to_ix)
        return tag_to_ix


class NERDataset(Dataset):
    def __init__(self, tokens_list, labels_list, word_to_ix, tag_to_ix):
        self.tokens_list = tokens_list
        self.labels_list = labels_list
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        labels = self.labels_list[idx]

        token_ids = [self.word_to_ix.get(tok, self.word_to_ix["<UNK>"]) for tok in tokens]
        label_ids = [self.tag_to_ix[label] for label in labels]

        return token_ids, label_ids


def pad_sequences(tensors_list):
    padded_tensor = []
    mask_tensor = []
    
    longest = len(max(tensors_list, key=len))

    for tensor in tensors_list:
        tensor = copy.deepcopy(tensor)
        mask = [1 for _ in tensor]
        while len(tensor) != longest:
            tensor.append(0)
            mask.append(0)
        padded_tensor.append(tensor)
        mask_tensor.append(mask)

    return torch.tensor(padded_tensor), torch.tensor(mask_tensor)


def collate_fn(batch):
    token_batch, label_batch = zip(*batch)
    input_tensor, input_mask = pad_sequences(token_batch)
    label_tensor, _ = pad_sequences(label_batch)
    return input_tensor, label_tensor, input_mask


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256, num_layers=1, dropout=0.1):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, input_tensor):
        embeds = self.embedding(input_tensor)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores


class NERTrainer:
    def __init__(self, model, train_loader, dev_loader, tag_to_ix, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.tag_to_ix = tag_to_ix
        self.device = device
        
        # Create reverse mapping for evaluation
        self.ix_to_tag = {v: k for k, v in tag_to_ix.items()}
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = nn.NLLLoss(ignore_index=0)  # Ignore <PAD> tokens
        
        self.train_losses = []
        self.f1_scores = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for input_tensor, label_tensor, input_mask in tqdm(self.train_loader, desc="Training"):
            input_tensor = input_tensor.to(self.device)
            label_tensor = label_tensor.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(input_tensor)
            
            # Flatten for loss calculation
            output_flat = output.view(-1, len(self.tag_to_ix))
            label_flat = label_tensor.view(-1)
            
            loss = self.loss_fn(output_flat, label_flat)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for input_tensor, label_tensor, input_mask in tqdm(self.dev_loader, desc="Evaluating"):
                input_tensor = input_tensor.to(self.device)
                label_tensor = label_tensor.to(self.device)
                input_mask = input_mask.to(self.device)
                
                output = self.model(input_tensor)
                preds = torch.argmax(output, dim=2)
                
                # Convert predictions and labels back to tags
                for i in range(input_tensor.size(0)):
                    pred_seq = []
                    label_seq = []
                    
                    for j in range(input_tensor.size(1)):
                        if input_mask[i][j] == 0:
                            continue  # Skip <PAD> tokens
                        
                        pred_tag = self.ix_to_tag[preds[i][j].item()]
                        true_tag = self.ix_to_tag[label_tensor[i][j].item()]
                        
                        pred_seq.append(pred_tag)
                        label_seq.append(true_tag)
                    
                    all_preds.append(pred_seq)
                    all_labels.append(label_seq)
        
        f1 = f1_score(all_labels, all_preds)
        self.f1_scores.append(f1)
        return f1
    
    def train(self, num_epochs=25, eval_every=5):
        print(f"Training for {num_epochs} epochs...")
        best_f1 = 0.0
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % eval_every == 0:
                f1 = self.evaluate()
                print(f"F1 Score: {f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    print(f"New best F1 score: {best_f1:.4f}")
        
        print(f"\nTraining completed. Best F1 score: {best_f1:.4f}")
        return best_f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_preprocessor = DataPreprocessor()
    
    train_dataset = NERDataset(data_preprocessor.train['tokens'], data_preprocessor.train['labels'], 
                              data_preprocessor.word_to_ix, data_preprocessor.tag_to_ix)
    dev_dataset = NERDataset(data_preprocessor.dev_x['tokens'], data_preprocessor.dev_y['labels'], 
                            data_preprocessor.word_to_ix, data_preprocessor.tag_to_ix)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    model = LSTMTagger(
        vocab_size=len(data_preprocessor.word_to_ix),
        tagset_size=len(data_preprocessor.tag_to_ix),
        embedding_dim=100,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = NERTrainer(model, train_loader, dev_loader, data_preprocessor.tag_to_ix, device)
    best_f1 = trainer.train(num_epochs=25, eval_every=5)
    
    return model, trainer, best_f1


if __name__ == "__main__":
    model, trainer, best_f1 = main()
