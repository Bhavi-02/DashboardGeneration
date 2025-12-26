<<<<<<< HEAD
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Dataset class
class PowerBINLUDataset(Dataset):
    def __init__(self, queries, intents, tokenizer, max_length=32):
        self.queries = queries
        self.intents = intents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = str(self.queries[idx])
        intent = self.intents[idx]

        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'intent': torch.tensor(intent, dtype=torch.long)
        }


# Model class
class IntentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(IntentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        dropped = self.dropout(pooled_output)
        return self.classifier(dropped)


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = 0
    correct_predictions = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['intent'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, targets)
        losses += loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), losses / len(data_loader)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['intent'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)
            losses += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)

    return correct_predictions.double() / len(data_loader.dataset), losses / len(data_loader)


def main():
    # Load dataset
    with open('data/training_data.json', 'r') as f:
        dataset = json.load(f)

    queries = [item['query'] for item in dataset]
    intents = [item['intent'] for item in dataset]

    # Encode intents to integers
    le = LabelEncoder()
    intents_encoded = le.fit_transform(intents)
    n_classes = len(le.classes_)

    # Split dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(queries, intents_encoded, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_dataset = PowerBINLUDataset(X_train, y_train, tokenizer)
    val_dataset = PowerBINLUDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IntentClassifier(n_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    epochs = 13

    best_accuracy = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')

        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device)
        print(f'Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'intent_classifier_model.bin')
            print('Model saved!')

    # Save the label encoder
    import pickle

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print('Training complete.')


if __name__ == '__main__':
    main()
=======
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Dataset class
class PowerBINLUDataset(Dataset):
    def __init__(self, queries, intents, tokenizer, max_length=32):
        self.queries = queries
        self.intents = intents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = str(self.queries[idx])
        intent = self.intents[idx]

        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'intent': torch.tensor(intent, dtype=torch.long)
        }


# Model class
class IntentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(IntentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        dropped = self.dropout(pooled_output)
        return self.classifier(dropped)


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = 0
    correct_predictions = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['intent'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, targets)
        losses += loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), losses / len(data_loader)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['intent'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)
            losses += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)

    return correct_predictions.double() / len(data_loader.dataset), losses / len(data_loader)


def main():
    # Load dataset
    with open('data/training_data.json', 'r') as f:
        dataset = json.load(f)

    queries = [item['query'] for item in dataset]
    intents = [item['intent'] for item in dataset]

    # Encode intents to integers
    le = LabelEncoder()
    intents_encoded = le.fit_transform(intents)
    n_classes = len(le.classes_)

    # Split dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(queries, intents_encoded, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_dataset = PowerBINLUDataset(X_train, y_train, tokenizer)
    val_dataset = PowerBINLUDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IntentClassifier(n_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    epochs = 13

    best_accuracy = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')

        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device)
        print(f'Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'intent_classifier_model.bin')
            print('Model saved!')

    # Save the label encoder
    import pickle

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print('Training complete.')


if __name__ == '__main__':
    main()
>>>>>>> chart-creator
