<<<<<<< HEAD
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import pickle
import sys

def load_ner_model():
    """Load the trained NER model"""
    model = DistilBertForTokenClassification.from_pretrained('ner_model')
    tokenizer = DistilBertTokenizerFast.from_pretrained('ner_model')
    
    with open('ner_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    model.eval()
    return model, tokenizer, label_encoder

def predict_entities(text, model, tokenizer, label_encoder, device):
    """Predict entities in text"""
    tokens = text.split()
    
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    word_ids = encoding.word_ids()
    predicted_labels = []
    
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id < len(tokens):
            label_id = predictions[0][i].cpu().numpy()
            label = label_encoder.inverse_transform([label_id])[0]
            if len(predicted_labels) <= word_id:
                predicted_labels.extend(['O'] * (word_id + 1 - len(predicted_labels)))
            predicted_labels[word_id] = label
    
    while len(predicted_labels) < len(tokens):
        predicted_labels.append('O')
    
    return tokens, predicted_labels[:len(tokens)]

def extract_entities(tokens, labels):
    """Extract structured entities"""
    entities = {'METRIC': [], 'DIMENSION': [], 'CHART_TYPE': [], 'FILTER': [], 'AGGREGATION': []}
    
    current_entity = None
    current_tokens = []
    current_type = None
    
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_entity:
                entities[current_type].append(' '.join(current_tokens))
            current_type = label[2:]
            current_tokens = [token]
            current_entity = True
        elif label.startswith('I-') and current_entity and label[2:] == current_type:
            current_tokens.append(token)
        else:
            if current_entity:
                entities[current_type].append(' '.join(current_tokens))
            current_entity = None
            current_tokens = []
            current_type = None
    
    if current_entity:
        entities[current_type].append(' '.join(current_tokens))
    
    return entities

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("🚀 Loading NER Model...")
        model, tokenizer, label_encoder = load_ner_model()
        model.to(device)
        print("✅ Model loaded!")
        
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
        else:
            query = input("💬 Enter your query: ")
        
        tokens, labels = predict_entities(query, model, tokenizer, label_encoder, device)
        entities = extract_entities(tokens, labels)
        
        print(f"\n📝 Query: {query}")
        print("\n🏷️  Entities Found:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"   {entity_type}: {', '.join(entity_list)}")
        
        print("\n🔤 Token Analysis:")
        for token, label in zip(tokens, labels):
            print(f"   {token} -> {label}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
=======
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import pickle
import sys

def load_ner_model():
    """Load the trained NER model"""
    model = DistilBertForTokenClassification.from_pretrained('ner_model')
    tokenizer = DistilBertTokenizerFast.from_pretrained('ner_model')
    
    with open('ner_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    model.eval()
    return model, tokenizer, label_encoder

def predict_entities(text, model, tokenizer, label_encoder, device):
    """Predict entities in text"""
    tokens = text.split()
    
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    word_ids = encoding.word_ids()
    predicted_labels = []
    
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id < len(tokens):
            label_id = predictions[0][i].cpu().numpy()
            label = label_encoder.inverse_transform([label_id])[0]
            if len(predicted_labels) <= word_id:
                predicted_labels.extend(['O'] * (word_id + 1 - len(predicted_labels)))
            predicted_labels[word_id] = label
    
    while len(predicted_labels) < len(tokens):
        predicted_labels.append('O')
    
    return tokens, predicted_labels[:len(tokens)]

def extract_entities(tokens, labels):
    """Extract structured entities"""
    entities = {'METRIC': [], 'DIMENSION': [], 'CHART_TYPE': [], 'FILTER': [], 'AGGREGATION': []}
    
    current_entity = None
    current_tokens = []
    current_type = None
    
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_entity:
                entities[current_type].append(' '.join(current_tokens))
            current_type = label[2:]
            current_tokens = [token]
            current_entity = True
        elif label.startswith('I-') and current_entity and label[2:] == current_type:
            current_tokens.append(token)
        else:
            if current_entity:
                entities[current_type].append(' '.join(current_tokens))
            current_entity = None
            current_tokens = []
            current_type = None
    
    if current_entity:
        entities[current_type].append(' '.join(current_tokens))
    
    return entities

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("🚀 Loading NER Model...")
        model, tokenizer, label_encoder = load_ner_model()
        model.to(device)
        print("✅ Model loaded!")
        
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
        else:
            query = input("💬 Enter your query: ")
        
        tokens, labels = predict_entities(query, model, tokenizer, label_encoder, device)
        entities = extract_entities(tokens, labels)
        
        print(f"\n📝 Query: {query}")
        print("\n🏷️  Entities Found:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"   {entity_type}: {', '.join(entity_list)}")
        
        print("\n🔤 Token Analysis:")
        for token, label in zip(tokens, labels):
            print(f"   {token} -> {label}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
>>>>>>> chart-creator
