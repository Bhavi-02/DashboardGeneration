#!/usr/bin/env python3
"""
Interactive Dashboard Pipeline
Simple flow: User Query → Chart Generation → Dynamic Dashboard
Modern ArchitectUI-inspired design without extra complexity
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from  dashboard_generator import DashboardGenerator
from nlu.chart_pipeline import NLUChartPipeline


class InteractiveDashboard:
    """
    Simplified dashboard pipeline with ArchitectUI design.
    User queries create charts that dynamically appear in a modern dashboard.
    """
    
    def __init__(self):
        """Initialize the interactive dashboard system"""
        print("🎨 Initializing Interactive Dashboard System...")
        print("="*60)
        
        try:
            self.dashboard = DashboardGenerator()
            print("✅ Dashboard generator ready")
        except Exception as e:
            print(f"❌ Dashboard initialization failed: {e}")
            self.dashboard = None
        
        try:
            self.nlu_pipeline = NLUChartPipeline()
            print("✅ NLU pipeline ready")
        except Exception as e:
            print(f"❌ NLU pipeline initialization failed: {e}")
            self.nlu_pipeline = None
    
    def add_chart_from_query(self, query):
        """
        Process a user query and add the resulting chart to the dashboard
        
        Args:
            query: Natural language query string
            
        Returns:
            True if successful, False otherwise
        """
        if not self.nlu_pipeline:
            print("❌ NLU pipeline not available")
            return False
        
        try:
            print(f"\n{'='*60}")
            print(f"📝 Processing: {query}")
            print(f"{'='*60}")
            
            # Process query through NLU pipeline
            result = self.nlu_pipeline.process_query(query)
            
            if result and result[0]:  # result is (fig, entities, tokens, labels)
                fig = result[0]
                entities = result[1]
                
                # Add chart to dashboard
                self.dashboard.add_chart_from_query(query, entities)
                print(f"✅ Chart added to dashboard! Total charts: {self.dashboard.get_chart_count()}")
                return True
            else:
                print("❌ Failed to generate chart from query")
                return False
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return False
    
    def generate_and_save_dashboard(self, filename="interactive_dashboard.html", title="Analytics Dashboard"):
        """
        Generate the final dashboard with all added charts
        
        Args:
            filename: Output HTML filename
            title: Dashboard title
            
        Returns:
            Plotly figure object or None
        """
        if not self.dashboard:
            print("❌ Dashboard generator not available")
            return None
        
        chart_count = self.dashboard.get_chart_count()
        
        if chart_count == 0:
            print("⚠️  No charts to display. Add some queries first!")
            return None
        
        print(f"\n{'='*60}")
        print(f"🎨 Generating dashboard with {chart_count} charts...")
        print(f"{'='*60}")
        
        try:
            fig = self.dashboard.generate_dashboard(
                title=title,
                save_path=filename
            )
            
            if fig:
                print(f"\n🎉 SUCCESS! Dashboard created with {chart_count} visualizations")
                print(f"📂 Saved to: {os.path.abspath(filename)}")
                return fig
            else:
                print("❌ Failed to generate dashboard")
                return None
                
        except Exception as e:
            print(f"❌ Error generating dashboard: {e}")
            return None
    
    def interactive_mode(self):
        """
        Interactive mode: Add multiple queries and generate dashboard
        """
        print("\n" + "="*60)
        print("🚀 INTERACTIVE DASHBOARD BUILDER")
        print("="*60)
        print("\n💡 How it works:")
        print("   1. Enter natural language queries about your data")
        print("   2. Charts are generated and added to your dashboard")
        print("   3. Type 'done' to generate and view the complete dashboard")
        print("   4. Type 'clear' to start over")
        print("   5. Type 'quit' to exit")
        print("\n📊 Example queries (based on Dataset.xlsx):")
        print("   • 'Show ExtendedAmount by SalesTerritoryKey'")
        print("   • 'Display OrderQuantity by ProductKey'")
        print("   • 'Show UnitPrice by ProductKey as bar chart'")
        print("   • 'Display Freight by SalesTerritoryKey as pie chart'")
        print("   • 'Show TaxAmt by CustomerKey top 10'")
        print("\n" + "="*60)
        
        while True:
            try:
                print(f"\n📊 Current charts: {self.dashboard.get_chart_count()}")
                query = input("\n💬 Enter query (or 'done'/'clear'/'quit'): ").strip()
                
                if not query:
                    continue
                
                query_lower = query.lower()
                
                if query_lower in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif query_lower in ['done', 'finish', 'generate']:
                    if self.dashboard.get_chart_count() == 0:
                        print("⚠️  Add at least one chart before generating dashboard!")
                        continue
                    
                    # Generate dashboard
                    fig = self.generate_and_save_dashboard()
                    
                    if fig:
                        # Try to open in browser
                        try:
                            import webbrowser
                            abs_path = os.path.abspath("interactive_dashboard.html")
                            webbrowser.open(f'file://{abs_path}')
                            print("🌐 Dashboard opened in your browser!")
                        except Exception as e:
                            print(f"⚠️  Could not auto-open browser: {e}")
                            print("📂 Please open interactive_dashboard.html manually")
                    
                    # Ask if they want to continue
                    cont = input("\n🔄 Continue adding more charts? (y/n): ").strip().lower()
                    if cont not in ['y', 'yes']:
                        print("\n👋 Goodbye!")
                        break
                
                elif query_lower in ['clear', 'reset', 'restart']:
                    self.dashboard.clear_charts()
                    print("✅ Dashboard cleared! Start fresh.")
                
                elif query_lower in ['summary', 'status', 'info']:
                    self.dashboard.print_summary()
                
                else:
                    # Process the query
                    self.add_chart_from_query(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_mode(self, queries, output_file="batch_dashboard.html", title="Batch Analytics Dashboard"):
        """
        Process multiple queries at once and generate dashboard
        
        Args:
            queries: List of query strings
            output_file: Output HTML filename
            title: Dashboard title
        """
        print(f"\n🚀 BATCH MODE: Processing {len(queries)} queries...")
        print("="*60)
        
        successful = 0
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing query...")
            if self.add_chart_from_query(query):
                successful += 1
        
        print(f"\n✅ Successfully processed {successful}/{len(queries)} queries")
        
        if successful > 0:
            return self.generate_and_save_dashboard(output_file, title)
        else:
            print("❌ No charts generated. Dashboard not created.")
            return None


def main():
    """Main function"""
    print("🎨 Interactive Dashboard with ArchitectUI Design")
    print("="*60)
    
    try:
        dashboard_system = InteractiveDashboard()
        
        if not dashboard_system.dashboard or not dashboard_system.nlu_pipeline:
            print("\n❌ System initialization failed!")
            print("💡 Make sure you have:")
            print("   • Trained the NER model (run train_ner.py)")
            print("   • Sample data available")
            return
        
        # Check command line arguments
        if len(sys.argv) > 1:
            # Batch mode with command line queries
            queries = [" ".join(sys.argv[1:])]
            dashboard_system.batch_mode(queries, "quick_dashboard.html", "Quick Dashboard")
        else:
            # Interactive mode
            dashboard_system.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()