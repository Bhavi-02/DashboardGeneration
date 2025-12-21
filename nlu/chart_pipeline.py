import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from charts.chart_generator import ChartGenerator
from nlu.smart_query_parser import SmartQueryParser

class NLUChartPipeline:
    """Complete pipeline from natural language query to chart generation"""
    
    def __init__(self, use_llm=True):
        """Initialize the NLU and chart generation pipeline"""
        self.chart_generator = ChartGenerator()
        
        # Initialize smart query parser (replaces NER model)
        self.smart_parser = SmartQueryParser(use_llm=use_llm)
        print("✅ Smart Query Parser initialized (Fuzzy + LLM hybrid)")
    
    def process_query(self, query):
        """Complete processing: Smart Parser -> Entity Extraction -> Chart Generation"""
        print(f"\n📝 Processing Query: {query}")
        print("=" * 60)
        
        # Step 1: Get available columns from data connector
        print("🔍 Step 1: Extracting available columns...")
        columns_info = self.chart_generator.data_connector.extract_all_columns_info()
        
        # Flatten all columns from all tables
        all_columns = []
        for table_name, info in columns_info.items():
            all_columns.extend(info['all_columns'])
        
        print(f"   Found {len(all_columns)} columns across {len(columns_info)} tables")
        
        # Extract sample data (head 5) for LLM context
        column_samples = {}
        try:
            for table_name, df in self.chart_generator.data_connector.cached_data.items():
                for col in df.columns:
                    if col not in column_samples:  # Avoid duplicates across tables
                        # Get first 5 non-null unique values
                        samples = df[col].dropna().unique()[:5].tolist()
                        column_samples[col] = samples
        except Exception as e:
            print(f"   ⚠️ Could not extract sample data: {e}")
            column_samples = None
        
        # Step 2: Parse query using smart parser with sample data
        print("🤖 Step 2: Parsing query with smart parser...")
        entities = self.smart_parser.parse_query(query, all_columns, column_samples)
        
        # Convert to old entity format for backward compatibility
        entities_compatible = {
            'METRIC': [entities['metric']] if entities['metric'] else [],
            'DIMENSION': [entities['dimension']] if entities['dimension'] else [],
            'CHART_TYPE': [entities['chart_type']] if entities['chart_type'] else ['bar'],
            'AGGREGATION': [entities['aggregation']] if entities['aggregation'] else ['sum'],
            'FILTER': entities.get('filters', [])
        }
        
        # Print extracted entities
        print("📋 Extracted Entities:")
        for entity_type, entity_list in entities_compatible.items():
            if entity_list:
                print(f"   {entity_type:12}: {', '.join(str(e) for e in entity_list)}")
            else:
                print(f"   {entity_type:12}: None")
        
        print(f"   Confidence  : {entities.get('match_confidence', 0)}%")
        
        # Step 3: Generate chart
        print("\n📊 Step 3: Generating chart...")
        try:
            fig = self.chart_generator.generate_chart(entities_compatible, query)
            print("✅ Chart generated successfully!")
            return fig, entities_compatible, None, None
            
        except Exception as e:
            print(f"❌ Error generating chart: {e}")
            import traceback
            traceback.print_exc()
            return None, entities_compatible, None, None
    
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
                fig, entities, _, _ = self.process_query(query)
                
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
            fig, entities, _, _ = pipeline.process_query(query)
            
            if fig:
                pipeline.save_and_show_chart(fig, "generated_chart", show=True)
        else:
            # Interactive mode
            pipeline.interactive_mode()
            
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        print("Make sure the OPENROUTER_API_KEY is set in .env for LLM fallback!")

if __name__ == "__main__":
    main()
