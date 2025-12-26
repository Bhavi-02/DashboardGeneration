<<<<<<< HEAD
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from seqeval.metrics import classification_report, f1_score
import pickle

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize and align labels
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Special token
            else:
                aligned_labels.append(labels[word_id])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

def prepare_data(data_path):
    """Load and prepare the NER data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['tokens'] for item in data]
    labels = [item['labels'] for item in data]
    
    # Create label encoder
    all_labels = set()
    for label_seq in labels:
        all_labels.update(label_seq)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_labels))
    
    # Encode labels
    encoded_labels = []
    for label_seq in labels:
        encoded_labels.append([label_encoder.transform([label])[0] for label in label_seq])
    
    return texts, encoded_labels, label_encoder

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, label_encoder, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            # Get predictions
            logits = outputs.logits
            pred_labels = torch.argmax(logits, dim=2)
            
            # Convert to lists and filter out special tokens
            for i in range(input_ids.shape[0]):
                pred_seq = []
                true_seq = []
                
                for j in range(input_ids.shape[1]):
                    if labels[i][j] != -100:  # Not a special token
                        pred_seq.append(label_encoder.inverse_transform([pred_labels[i][j].cpu().numpy()])[0])
                        true_seq.append(label_encoder.inverse_transform([labels[i][j].cpu().numpy()])[0])
                
                if pred_seq and true_seq:  # Only add non-empty sequences
                    predictions.append(pred_seq)
                    true_labels.append(true_seq)
    
    # Calculate metrics
    f1 = f1_score(true_labels, predictions)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, f1, predictions, true_labels

def main():
    print("🚀 Starting NER Model Training for Power BI Queries")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = 'data/annotate_data.json'
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load and prepare data
    print("📁 Loading data...")
    texts, labels, label_encoder = prepare_data(DATA_PATH)
    
    print(f"📊 Dataset Info:")
    print(f"   Total samples: {len(texts)}")
    print(f"   Unique labels: {len(label_encoder.classes_)}")
    print(f"   Label classes: {list(label_encoder.classes_)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Initialize tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = NERDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = NERDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("🤖 Initializing model...")
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("🎯 Starting training...")
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        print(f"\n📈 Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_f1, predictions, true_labels = evaluate_model(
            model, val_loader, label_encoder, device
        )
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("💾 Saving best model...")
            
            # Save model
            model.save_pretrained('ner_model')
            tokenizer.save_pretrained('ner_model')
            
            # Save label encoder
            with open('ner_label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
            
            print(f"✅ New best F1 score: {best_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    
    # Show detailed classification report for best model
    print("\n📋 Final Classification Report:")
    print(classification_report(true_labels, predictions))
    
    # Show example predictions
    print("\n🔍 Example Predictions:")
    print("-" * 40)
    for i in range(min(3, len(predictions))):
        print(f"True:  {' '.join(true_labels[i])}")
        print(f"Pred:  {' '.join(predictions[i])}")
        print("-" * 40)

if __name__ == "__main__":
    main()
=======
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from seqeval.metrics import classification_report, f1_score
import pickle

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize and align labels
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Special token
            else:
                aligned_labels.append(labels[word_id])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

def prepare_data(data_path):
    """Load and prepare the NER data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['tokens'] for item in data]
    labels = [item['labels'] for item in data]
    
    # Create label encoder
    all_labels = set()
    for label_seq in labels:
        all_labels.update(label_seq)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_labels))
    
    # Encode labels
    encoded_labels = []
    for label_seq in labels:
        encoded_labels.append([label_encoder.transform([label])[0] for label in label_seq])
    
    return texts, encoded_labels, label_encoder

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, label_encoder, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            # Get predictions
            logits = outputs.logits
            pred_labels = torch.argmax(logits, dim=2)
            
            # Convert to lists and filter out special tokens
            for i in range(input_ids.shape[0]):
                pred_seq = []
                true_seq = []
                
                for j in range(input_ids.shape[1]):
                    if labels[i][j] != -100:  # Not a special token
                        pred_seq.append(label_encoder.inverse_transform([pred_labels[i][j].cpu().numpy()])[0])
                        true_seq.append(label_encoder.inverse_transform([labels[i][j].cpu().numpy()])[0])
                
                if pred_seq and true_seq:  # Only add non-empty sequences
                    predictions.append(pred_seq)
                    true_labels.append(true_seq)
    
    # Calculate metrics
    f1 = f1_score(true_labels, predictions)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, f1, predictions, true_labels

def main():
    print("🚀 Starting NER Model Training for Power BI Queries")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = 'data/annotate_data.json'
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load and prepare data
    print("📁 Loading data...")
    texts, labels, label_encoder = prepare_data(DATA_PATH)
    
    print(f"📊 Dataset Info:")
    print(f"   Total samples: {len(texts)}")
    print(f"   Unique labels: {len(label_encoder.classes_)}")
    print(f"   Label classes: {list(label_encoder.classes_)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Initialize tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = NERDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = NERDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("🤖 Initializing model...")
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("🎯 Starting training...")
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        print(f"\n📈 Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_f1, predictions, true_labels = evaluate_model(
            model, val_loader, label_encoder, device
        )
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("💾 Saving best model...")
            
            # Save model
            model.save_pretrained('ner_model')
            tokenizer.save_pretrained('ner_model')
            
            # Save label encoder
            with open('ner_label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
            
            print(f"✅ New best F1 score: {best_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    
    # Show detailed classification report for best model
    print("\n📋 Final Classification Report:")
    print(classification_report(true_labels, predictions))
    
    # Show example predictions
    print("\n🔍 Example Predictions:")
    print("-" * 40)
    for i in range(min(3, len(predictions))):
        print(f"True:  {' '.join(true_labels[i])}")
        print(f"Pred:  {' '.join(predictions[i])}")
        print("-" * 40)

if __name__ == "__main__":
    main()
>>>>>>> chart-creator
