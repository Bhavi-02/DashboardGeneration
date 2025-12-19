import torch
from transformers import DistilBertTokenizerFast
import pickle

# Define the same model class as used during training
import torch.nn as nn
from transformers import DistilBertModel

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

def load_model_and_encoder(model_path, encoder_path, device):
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    model = IntentClassifier(len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, label_encoder

def predict_intent(model, label_encoder, tokenizer, query, device):
    encoding = tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        intent = label_encoder.inverse_transform([pred_idx])[0]
    return intent, confidence

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and encoder
    model, label_encoder = load_model_and_encoder('intent_classifier_model.bin', 'label_encoder.pkl', device)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Test queries
    test_queries = [
        "Show me sales by region in a bar chart",
        "Filter to show 2024 data only",
        "What are the top products by sales?",
        "Create a pie chart for customer distribution"
    ]


    for query in test_queries:
        intent, confidence = predict_intent(model, label_encoder, tokenizer, query, device)
        print(f"Query: {query}")
        print(f"Predicted Intent: {intent} (Confidence: {confidence:.3f})")
        print("-" * 50)

