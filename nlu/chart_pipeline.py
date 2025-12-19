import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from charts.chart_generator import ChartGenerator

class NLUChartPipeline:
    """Complete pipeline from natural language query to chart generation"""
    
    def __init__(self, ner_model_path='ner_model', ner_encoder_path='ner_label_encoder.pkl'):
        """Initialize the NLU and chart generation pipeline"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ner_model = None
        self.ner_tokenizer = None
        self.ner_label_encoder = None
        self.chart_generator = ChartGenerator()
        
        # Load NER model
        self.load_ner_model(ner_model_path, ner_encoder_path)
        
    def load_ner_model(self, model_path, encoder_path):
        """Load the trained NER model"""
        try:
            print("🤖 Loading NER model...")
            self.ner_model = DistilBertForTokenClassification.from_pretrained(model_path)
            self.ner_tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            
            with open(encoder_path, 'rb') as f:
                self.ner_label_encoder = pickle.load(f)
            
            self.ner_model.to(self.device)
            self.ner_model.eval()
            print("✅ NER model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading NER model: {e}")
            raise
    
    def predict_entities(self, text):
        """Extract entities from text using NER model"""
        tokens = text.split()
        
        encoding = self.ner_tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.ner_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        word_ids = encoding.word_ids()
        predicted_labels = []
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(tokens):
                label_id = predictions[0][i].cpu().numpy()
                label = self.ner_label_encoder.inverse_transform([label_id])[0]
                # Ensure the list is long enough before assignment
                while len(predicted_labels) <= word_id:
                    predicted_labels.append('O')
                predicted_labels[word_id] = label
        
        while len(predicted_labels) < len(tokens):
            predicted_labels.append('O')
        
        return tokens, predicted_labels[:len(tokens)]
    
    def extract_structured_entities(self, tokens, labels):
        """Convert token-level labels to structured entities"""
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
                # Save previous entity
                if current_entity:
                    entities[current_type].append(' '.join(current_tokens))
                
                # Start new entity
                current_type = label[2:]
                current_tokens = [token]
                current_entity = True
                
            elif label.startswith('I-') and current_entity and label[2:] == current_type:
                # Continue current entity
                current_tokens.append(token)
                
            else:
                # End current entity
                if current_entity:
                    entities[current_type].append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []
                current_type = None
        
        # Don't forget the last entity
        if current_entity:
            entities[current_type].append(' '.join(current_tokens))
        
        return entities
    
    def process_query(self, query):
        """Complete processing: NLU -> Entity Extraction -> Chart Generation"""
        print(f"\n📝 Processing Query: {query}")
        print("=" * 60)
        
        # Step 1: Extract entities using NER
        print("🔍 Step 1: Extracting entities...")
        tokens, labels = self.predict_entities(query)
        entities = self.extract_structured_entities(tokens, labels)
        
        # Print extracted entities
        print("📋 Extracted Entities:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"   {entity_type:12}: {', '.join(entity_list)}")
            else:
                print(f"   {entity_type:12}: None")
        
        # Step 2: Generate chart
        print("\n📊 Step 2: Generating chart...")
        try:
            fig = self.chart_generator.generate_chart(entities, query)
            print("✅ Chart generated successfully!")
            return fig, entities, tokens, labels
            
        except Exception as e:
            print(f"❌ Error generating chart: {e}")
            return None, entities, tokens, labels
    
    def save_and_show_chart(self, fig, filename=None, show=True, save_format='html'):
        """Save and/or display the generated chart"""
        if fig is None:
            print("❌ No chart to display")
            return
            
        # Save chart
        if filename:
            self.chart_generator.save_chart(fig, filename, format=save_format)
            print(f"💾 Chart saved as {filename}.{save_format}")
        
        # Show chart
        if show:
            self.chart_generator.show_chart(fig)
    
    def interactive_mode(self):
        """Interactive mode for testing queries"""
        print("🚀 NLU Chart Generation Pipeline")
        print("=" * 50)
        print("Enter natural language queries to generate charts!")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        chart_count = 0
        
        while True:
            try:
                query = input("\n💬 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not query:
                    continue
                
                # Process query
                fig, entities, tokens, labels = self.process_query(query)
                
                if fig:
                    chart_count += 1
                    filename = f"chart_{chart_count}"
                    
                    # Ask user if they want to save
                    save_choice = input(f"\n💾 Save chart as {filename}.html? (y/n): ").strip().lower()
                    save_file = filename if save_choice in ['y', 'yes'] else None
                    
                    self.save_and_show_chart(fig, save_file, show=True)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    try:
        # Initialize pipeline
        pipeline = NLUChartPipeline()
        
        if len(sys.argv) > 1:
            # Command line mode
            query = " ".join(sys.argv[1:])
            fig, entities, tokens, labels = pipeline.process_query(query)
            
            if fig:
                pipeline.save_and_show_chart(fig, "generated_chart", show=True)
        else:
            # Interactive mode
            pipeline.interactive_mode()
            
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        print("Make sure you have trained the NER model first!")

if __name__ == "__main__":
    main()
