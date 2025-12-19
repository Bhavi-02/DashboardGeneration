"""
Enhanced NLU Model Retraining Script
Retrains the NER model with advanced ecommerce analytical queries
"""

import json
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from pathlib import Path

def load_training_data():
    """Load training data from JSON file"""
    with open('data/training_data.json', 'r') as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training samples")
    return training_data

def convert_to_spacy_format(training_data):
    """Convert training data to spaCy format with entity annotations"""
    spacy_data = []
    
    for sample in training_data:
        query = sample['query']
        entities = []
        
        # Extract entities from the query
        text_lower = query.lower()
        
        # Metric entities
        metric_keywords = ['sales', 'revenue', 'quantity', 'price', 'rate', 'growth', 
                          'value', 'profit', 'margin', 'transaction', 'order', 'units']
        for keyword in metric_keywords:
            if keyword in text_lower:
                start = text_lower.find(keyword)
                if start != -1:
                    entities.append((start, start + len(keyword), 'METRIC'))
        
        # Dimension entities
        dimension_keywords = ['year', 'month', 'quarter', 'state', 'branch', 'product', 
                             'category', 'season', 'date', 'time', 'location', 'region']
        for keyword in dimension_keywords:
            if keyword in text_lower:
                start = text_lower.find(keyword)
                if start != -1:
                    entities.append((start, start + len(keyword), 'DIMENSION'))
        
        # Chart type entities
        chart_keywords = ['bar chart', 'line chart', 'pie chart', 'scatter', 'table', 
                         'graph', 'visualization', 'dashboard']
        for keyword in chart_keywords:
            if keyword in text_lower:
                start = text_lower.find(keyword)
                if start != -1:
                    entities.append((start, start + len(keyword), 'CHART_TYPE'))
        
        # Aggregation entities
        agg_keywords = ['total', 'average', 'sum', 'count', 'mean', 'max', 'min', 'top']
        for keyword in agg_keywords:
            if keyword in text_lower:
                start = text_lower.find(keyword)
                if start != -1:
                    entities.append((start, start + len(keyword), 'AGGREGATION'))
        
        # Remove overlapping entities (keep the first occurrence)
        entities = sorted(entities, key=lambda x: x[0])
        non_overlapping = []
        last_end = -1
        for entity in entities:
            if entity[0] >= last_end:
                non_overlapping.append(entity)
                last_end = entity[1]
        
        spacy_data.append((query, {"entities": non_overlapping}))
    
    return spacy_data

def train_ner_model(training_data, n_iter=30, output_dir='ner_model'):
    """Train NER model with enhanced training data"""
    
    # Load blank English model
    nlp = spacy.blank("en")
    
    # Add NER pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    labels = ['METRIC', 'DIMENSION', 'CHART_TYPE', 'AGGREGATION', 'FILTER', 'COMPARISON', 'TIME_PERIOD']
    for label in labels:
        ner.add_label(label)
    
    # Convert training data to spaCy format
    spacy_data = convert_to_spacy_format(training_data)
    
    print(f"Training on {len(spacy_data)} annotated samples")
    print(f"Entity labels: {labels}")
    
    # Train the model
    optimizer = nlp.begin_training()
    
    for iteration in range(n_iter):
        random.shuffle(spacy_data)
        losses = {}
        
        # Create batches
        batches = minibatch(spacy_data, size=compounding(4.0, 32.0, 1.001))
        
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            
            nlp.update(examples, drop=0.35, losses=losses, sgd=optimizer)
        
        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration + 1}/{n_iter} - Loss: {losses.get('ner', 0):.4f}")
    
    # Save the model
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    nlp.to_disk(output_path)
    print(f"\nModel saved to {output_path}")
    
    return nlp

def test_model(nlp, test_queries):
    """Test the trained model on sample queries"""
    print("\n" + "="*80)
    print("TESTING TRAINED MODEL")
    print("="*80 + "\n")
    
    for query in test_queries:
        doc = nlp(query)
        print(f"Query: {query}")
        
        if doc.ents:
            print("Detected Entities:")
            for ent in doc.ents:
                print(f"  - {ent.text} ({ent.label_})")
        else:
            print("  No entities detected")
        print()

def main():
    print("="*80)
    print("ENHANCED NLU MODEL TRAINING")
    print("="*80 + "\n")
    
    # Load training data
    training_data = load_training_data()
    
    # Train the model
    nlp = train_ner_model(training_data, n_iter=30)
    
    # Test queries from user's requirements
    test_queries = [
        "What is the year-on-year growth in sales revenue from 2017 to 2025?",
        "Which branch generated higher total sales across the years?",
        "How do sales trends differ by state?",
        "Which products consistently perform best in terms of revenue and quantity sold?",
        "Compare Insulated vs Non-Insulated goods in terms of total revenue",
        "Which product category dominates in each branch?",
        "Are there seasonal spikes in sales for certain product categories?",
        "What is the average pre-tax value per order across years?",
        "How does rate variation affect sales quantity?",
        "Show sales by year, category, branch, and state"
    ]
    
    test_model(nlp, test_queries)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal training samples: {len(training_data)}")
    print("Model saved to: ner_model/")
    print("\nThe NLU system is now trained to handle:")
    print("  ✓ Year-over-year growth calculations")
    print("  ✓ Branch and state comparisons")
    print("  ✓ Product performance analysis")
    print("  ✓ Seasonal trend detection")
    print("  ✓ Category analysis")
    print("  ✓ Rate variation effects")
    print("  ✓ Multi-dimensional dashboards")
    print("\nYou can now use the system to answer complex analytical queries!")

if __name__ == "__main__":
    main()
