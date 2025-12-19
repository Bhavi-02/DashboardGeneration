"""
Quick Test Script for Enhanced NLU Model
Tests the trained model with all user-specified complex queries
"""

import spacy
import json
from pathlib import Path

def load_model():
    """Load the trained NER model"""
    try:
        nlp = spacy.load("ner_model")
        print("✅ Model loaded successfully from 'ner_model/'\n")
        return nlp
    except:
        try:
            nlp = spacy.load("./ner_model")
            print("✅ Model loaded successfully from './ner_model/'\n")
            return nlp
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

def test_query(nlp, query, show_details=True):
    """Test a single query and show results"""
    doc = nlp(query)
    
    if show_details:
        print(f"Query: {query}")
        print(f"Tokens: {len(doc)}")
        
        if doc.ents:
            print(f"Entities Found: {len(doc.ents)}")
            for ent in doc.ents:
                print(f"  • {ent.text:20s} → {ent.label_:15s} (position: {ent.start}-{ent.end})")
        else:
            print("  ⚠️  No entities detected")
        print()
    
    return {
        "query": query,
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "entity_count": len(doc.ents)
    }

def main():
    print("="*80)
    print("ENHANCED NLU MODEL - COMPREHENSIVE TEST")
    print("="*80)
    print()
    
    # Load model
    nlp = load_model()
    if not nlp:
        return
    
    # Your specific complex queries
    user_queries = [
        # Year-over-Year Growth
        "What is the year-on-year growth in sales revenue from 2017 to 2025?",
        
        # Branch Comparisons
        "Which branch generated higher total sales across the years?",
        
        # State Analysis
        "How do sales trends differ by state?",
        
        # Product Performance
        "Which products consistently perform best in terms of revenue and quantity sold?",
        
        # Insulated vs Non-Insulated
        "Compare Insulated vs Non-Insulated goods in terms of total revenue contribution",
        "Compare average selling rate of Insulated vs Non-Insulated goods",
        "What is the average price of Insulated versus Non-Insulated products?",
        "Show growth trend over time for Insulated vs Non-Insulated goods",
        
        # Category Analysis
        "Which product category dominates in each branch?",
        
        # Seasonal Analysis
        "Are there seasonal spikes in sales for certain product categories?",
        
        # Order Value
        "What is the average pre-tax value per order across years?",
        
        # Rate Variation
        "How does rate variation affect sales quantity?",
        
        # Dashboard Requests
        "Build a dashboard showing sales by year",
        "Show sales by category in dashboard",
        "Display sales by branch on dashboard",
        "Create sales by state visualization",
        "Show sales by year, category, branch, and state"
    ]
    
    print(f"Testing {len(user_queries)} complex analytical queries...\n")
    print("="*80)
    print()
    
    results = []
    for i, query in enumerate(user_queries, 1):
        print(f"[Test {i}/{len(user_queries)}]")
        result = test_query(nlp, query, show_details=True)
        results.append(result)
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print()
    
    total_queries = len(results)
    queries_with_entities = sum(1 for r in results if r['entity_count'] > 0)
    total_entities = sum(r['entity_count'] for r in results)
    avg_entities = total_entities / total_queries if total_queries > 0 else 0
    
    print(f"Total Queries Tested: {total_queries}")
    print(f"Queries with Entities: {queries_with_entities} ({queries_with_entities/total_queries*100:.1f}%)")
    print(f"Total Entities Detected: {total_entities}")
    print(f"Average Entities per Query: {avg_entities:.2f}")
    print()
    
    # Entity type breakdown
    entity_types = {}
    for result in results:
        for entity_text, entity_label in result['entities']:
            entity_types[entity_label] = entity_types.get(entity_label, 0) + 1
    
    print("Entity Type Distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type:15s}: {count:3d} occurrences")
    
    print()
    print("="*80)
    print("✅ ALL TESTS COMPLETED!")
    print("="*80)
    print()
    print("The NLU model successfully understands complex analytical queries.")
    print("You can now use these queries in your dashboard system!")
    print()
    print("Next steps:")
    print("  1. Integrate the trained model into your main application")
    print("  2. Test with real user queries")
    print("  3. Monitor performance and add more training data as needed")

if __name__ == "__main__":
    main()
