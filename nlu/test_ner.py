import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import pickle
import json

def load_ner_model(model_path, label_encoder_path):
    """Load the trained NER model and label encoder"""
    # Load model and tokenizer
    model = DistilBertForTokenClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    
    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    model.eval()
    return model, tokenizer, label_encoder

def predict_ner(text, model, tokenizer, label_encoder, device):
    """Predict NER labels for a given text"""
    # Tokenize input
    if isinstance(text, str):
        tokens = text.split()
    else:
        tokens = text
    
    # Encode text
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
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Align predictions with original tokens
    word_ids = encoding.word_ids()
    predicted_labels = []
    
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id < len(tokens):
            label_id = predictions[0][i].cpu().numpy()
            label = label_encoder.inverse_transform([label_id])[0]
            if len(predicted_labels) <= word_id:
                predicted_labels.extend(['O'] * (word_id + 1 - len(predicted_labels)))
            predicted_labels[word_id] = label
    
    # Ensure we have labels for all tokens
    while len(predicted_labels) < len(tokens):
        predicted_labels.append('O')
    
    return tokens, predicted_labels[:len(tokens)]

def extract_entities(tokens, labels):
    """Extract entities from tokens and labels"""
    entities = {
        'METRIC': [],
        'DIMENSION': [],
        'CHART_TYPE': [],
        'FILTER': [],
        'AGGREGATION': []
    }
    
    current_entity = None
    current_tokens = []
    current_type = None
    
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            # Start of new entity
            if current_entity:
                # Save previous entity
                entity_text = ' '.join(current_tokens)
                if current_type in entities:
                    entities[current_type].append(entity_text)
            
            current_type = label[2:]  # Remove 'B-' prefix
            current_tokens = [token]
            current_entity = True
            
        elif label.startswith('I-') and current_entity and label[2:] == current_type:
            # Continue current entity
            current_tokens.append(token)
            
        else:
            # End current entity
            if current_entity:
                entity_text = ' '.join(current_tokens)
                if current_type in entities:
                    entities[current_type].append(entity_text)
            
            current_entity = None
            current_tokens = []
            current_type = None
    
    # Don't forget the last entity
    if current_entity:
        entity_text = ' '.join(current_tokens)
        if current_type in entities:
            entities[current_type].append(entity_text)
    
    return entities

def print_ner_results(text, tokens, labels, entities):
    """Print formatted NER results"""
    print("\n" + "="*60)
    print("🔍 NER ANALYSIS")
    print("="*60)
    print(f"📝 Query: {text if isinstance(text, str) else ' '.join(text)}")
    print("\n🏷️  TOKEN LABELS:")
    for token, label in zip(tokens, labels):
        print(f"   {token:15} -> {label}")
    
    print("\n📋 EXTRACTED ENTITIES:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"   {entity_type:12}: {', '.join(entity_list)}")
        else:
            print(f"   {entity_type:12}: None")
    print("="*60)

def test_ner_model():
    """Test the NER model with sample queries"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("🚀 Loading NER Model...")
        model, tokenizer, label_encoder = load_ner_model('ner_model', 'ner_label_encoder.pkl')
        model.to(device)
        print("✅ Model loaded successfully!")
        
        # Test queries
        test_queries = [
            "Show me sales by region in a bar chart",
            "Create a line graph for revenue trends over last 12 months",
            "What are the top 10 products by profit margin",
            "Filter data to show only Q1 2024 results",
            "Display customer segmentation as a pie chart",
            "Show total sales for North region",
            "Create scatter plot comparing marketing spend vs revenue by quarter"
        ]
        
        print(f"\n🎯 Testing with {len(test_queries)} sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i} ---")
            tokens, labels = predict_ner(query, model, tokenizer, label_encoder, device)
            entities = extract_entities(tokens, labels)
            print_ner_results(query, tokens, labels, entities)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please train the model first.")
        print(f"   Missing: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_ner_test():
    """Interactive NER testing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("🚀 Loading NER Model for Interactive Testing...")
        model, tokenizer, label_encoder = load_ner_model('ner_model', 'ner_label_encoder.pkl')
        model.to(device)
        print("✅ Model loaded successfully!")
        
        print("\n💬 Enter your queries (type 'quit' to exit):")
        
        while True:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query:
                tokens, labels = predict_ner(query, model, tokenizer, label_encoder, device)
                entities = extract_entities(tokens, labels)
                print_ner_results(query, tokens, labels, entities)
            
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please train the model first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_ner_test()
    else:
        test_ner_model()
