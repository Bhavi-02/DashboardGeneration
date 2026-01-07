"""
Dashboard Explainability Module using AI/LLM
Provides intelligent insights and explanations for generated dashboards
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# Import LLM for intelligent explanations
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from dotenv import load_dotenv
    
    load_dotenv()
    LLM_AVAILABLE = True
    
    # Initialize LLM for chart explanations
    chart_llm = ChatOpenAI(
        model=os.getenv("GENERATIVE_MODEL", "anthropic/claude-3-haiku"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,  # Slightly creative but still focused
        max_tokens=800,   # Longer responses for detailed explanations
        top_p=0.95,
        default_headers={"HTTP-Referer": "http://localhost"}
    )
    print("✅ LLM initialized for chart explanations")
    
except Exception as e:
    print(f"⚠️ LLM not available for chart explanations: {e}")
    LLM_AVAILABLE = False
    chart_llm = None

# Import RAG components - with error handling
try:
    from rag.rag import (
        TfidfEmbeddings,
        SimpleVectorStore,
        process_questions_with_rag
    )
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG components not fully available: {e}")
    RAG_AVAILABLE = False
    # Create dummy classes
    class TfidfEmbeddings:
        def __init__(self, *args, **kwargs):
            pass
    class SimpleVectorStore:
        def __init__(self, *args, **kwargs):
            pass
    def process_questions_with_rag(*args, **kwargs):
        return []

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class DashboardExplainer:
    """
    RAG-based dashboard explainer that provides insights by analyzing:
    1. Company profile document
    2. Dataset information
    3. Generated dashboard charts
    """
    
    def __init__(self):
        self.embeddings = TfidfEmbeddings(max_features=500)
        self.company_docs = None
        self.dataset_docs = None
        self.vectorstore = None
        self.company_profile_path = None
        self.dataset_path = None
        
    def load_company_profile(self, file_path_or_url: str) -> bool:
        """
        Load company profile document (simple text loading only)
        """
        try:
            print(f"📄 Loading company profile from: {file_path_or_url}")
            
            content = ""
            
            # Check if it's a local file
            if os.path.exists(file_path_or_url):
                # Local file
                with open(file_path_or_url, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path_or_url.startswith('http'):
                # Simple URL text download
                import requests
                response = requests.get(file_path_or_url)
                content = response.text
            else:
                # Treat as direct text input
                content = file_path_or_url
            
            # Create document chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(content)
            
            self.company_docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": "company_profile",
                        "type": "company_profile",
                        "chunk": i
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            self.company_profile_path = "company_profile"
            print(f"✅ Company profile loaded: {len(self.company_docs)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error loading company profile: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_dataset_info(self, dataset_info: Dict[str, Any]) -> bool:
        """
        Load dataset information (schema, statistics, sample data)
        """
        try:
            print("📊 Loading dataset information...")
            
            # Convert dataset info to text documents
            dataset_text = self._format_dataset_info(dataset_info)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )
            chunks = text_splitter.split_text(dataset_text)
            
            self.dataset_docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": "dataset_schema",
                        "type": "dataset_info",
                        "chunk": i
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            print(f"✅ Dataset information loaded: {len(self.dataset_docs)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset info: {e}")
            return False
    
    def _format_dataset_info(self, dataset_info: Dict[str, Any]) -> str:
        """
        Format dataset information into readable text for RAG processing
        """
        text_parts = ["DATASET INFORMATION\n" + "="*50 + "\n"]
        
        tables = dataset_info.get('tables', [])
        for table in tables:
            table_name = table.get('name', 'Unknown')
            row_count = table.get('row_count', 0)
            
            text_parts.append(f"\nTable: {table_name}")
            text_parts.append(f"Total Records: {row_count:,}")
            text_parts.append("\nColumns:")
            
            for col in table.get('columns', []):
                col_name = col.get('name')
                col_type = col.get('type')
                text_parts.append(f"  - {col_name} ({col_type})")
            
            text_parts.append("\n" + "-"*50)
        
        return "\n".join(text_parts)
    
    def initialize_rag_system(self) -> bool:
        """
        Initialize RAG system with loaded documents
        """
        try:
            if not self.company_docs and not self.dataset_docs:
                print("⚠️ No documents loaded. Please load company profile and dataset first.")
                return False
            
            print("🔧 Initializing RAG system...")
            
            # Combine all documents
            all_docs = []
            if self.company_docs:
                all_docs.extend(self.company_docs)
            if self.dataset_docs:
                all_docs.extend(self.dataset_docs)
            
            # Create vector store
            if FAISS_AVAILABLE:
                self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
                print("✅ FAISS vector store created")
            else:
                self.vectorstore = SimpleVectorStore(all_docs, self.embeddings)
                print("✅ Simple vector store created")
            
            print(f"✅ RAG system initialized with {len(all_docs)} document chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False
    
    def explain_dashboard(self, dashboard_config: Dict[str, Any], force_chart_only: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a dashboard
        
        Args:
            dashboard_config: Dictionary containing dashboard charts and metadata
            force_chart_only: If True, use chart-only mode even if documents are loaded
            
        Returns:
            Dictionary with explanations for each chart and overall insights
        """
        try:
            # Extract charts from config
            charts = dashboard_config.get('charts', [])
            
            if not charts:
                return {
                    "success": False,
                    "error": "No charts found in dashboard configuration"
                }
            
            # Check if we should use chart-only mode
            has_documents = bool(self.company_docs or self.dataset_docs)
            
            if force_chart_only or not has_documents:
                # Force chart-only mode or no documents provided
                mode_reason = "forced by request" if force_chart_only else "no documents provided"
                print(f"📊 Generating chart-only explanations ({mode_reason})...")
                return self._explain_charts_only(dashboard_config, charts)
            
            # Documents provided - use RAG + chart insights
            if not self.vectorstore:
                return {
                    "success": False,
                    "error": "RAG system not initialized. Please initialize with documents first."
                }
            
            print("🔍 Generating RAG-enhanced dashboard explanations...")
            
            # Create chart documents for RAG context
            chart_docs = self._create_chart_documents(charts)
            
            # Add chart documents to vectorstore temporarily
            all_docs = []
            if self.company_docs:
                all_docs.extend(self.company_docs)
            if self.dataset_docs:
                all_docs.extend(self.dataset_docs)
            all_docs.extend(chart_docs)
            
            # Generate questions for each chart
            questions = self._generate_chart_questions(charts)
            
            # Add overall business insights questions
            questions.extend([
                "Based on the uploaded file, dataset, and generated charts, what are the key business insights?",
                "What trends or patterns are most significant for decision-making across all visualizations?",
                "What recommendations can be made based on this data and the company profile?",
                "Are there any potential risks or opportunities identified in the data?"
            ])
            
            # Process questions using RAG
            print(f"📝 Processing {len(questions)} explanation questions...")
            answers = process_questions_with_rag(self.vectorstore, all_docs, questions)
            
            # Format the results
            chart_explanations = []
            num_charts = len(charts)
            
            for i, chart in enumerate(charts):
                query = chart.get('query', f'Chart {i+1}')
                chart_type = self._infer_chart_type(chart)
                
                # Parse measure and dimension from query
                measure, dimension = self._parse_query_components(query)
                
                # Get the RAG-generated explanation
                rag_explanation = answers[i] if i < len(answers) else ""
                
                # Build structured explanation
                full_explanation = ""
                
                # Add chart type description
                if chart_type == 'bar':
                    full_explanation += "Chart Type: Bar Chart\n"
                    full_explanation += "A bar chart uses rectangular bars to compare values across categories. "
                elif chart_type == 'line':
                    full_explanation += "Chart Type: Line Chart\n"
                    full_explanation += "A line chart connects data points to show trends over time or categories. "
                elif chart_type == 'pie':
                    full_explanation += "Chart Type: Pie Chart\n"
                    full_explanation += "A pie chart shows proportions as slices of a circular diagram. "
                elif chart_type == 'scatter':
                    full_explanation += "Chart Type: Scatter Plot\n"
                    full_explanation += "A scatter plot displays relationships between two variables. "
                else:
                    full_explanation += "Chart Type: Visualization\n"
                
                full_explanation += "\n\n"
                
                # Add axis information
                if measure and dimension:
                    full_explanation += f"X-Axis (Horizontal): {dimension}\n"
                    full_explanation += f"Y-Axis (Vertical): {measure}\n\n"
                    full_explanation += f"Relationship: The chart shows how {measure} varies across different {dimension}. "
                    full_explanation += f"Each point or bar represents a unique {dimension} value, allowing you to compare {measure} between categories.\n\n"
                
                # Add RAG-generated insights
                if rag_explanation and rag_explanation != "No explanation available":
                    full_explanation += "Business Insights:\n"
                    full_explanation += rag_explanation + "\n"
                else:
                    full_explanation += "Business Insights:\n"
                    full_explanation += f"This visualization helps analyze {measure if measure else 'the data'} "
                    full_explanation += f"across {dimension if dimension else 'different categories'}. "
                    full_explanation += "Users can identify top performers, trends, outliers, and make data-driven decisions.\n"
                
                explanation = {
                    "chart_id": i + 1,
                    "chart_title": query,
                    "chart_type": chart_type,
                    "chart_type_display": chart_type.title() + " Chart" if chart_type != 'unknown' else "Visualization",
                    "x_axis": dimension if dimension else "Category",
                    "y_axis": measure if measure else "Value",
                    "query": query,
                    "explanation": full_explanation,
                    "insights": self._extract_insights(full_explanation)
                }
                chart_explanations.append(explanation)
            
            # Overall insights from the remaining answers
            overall_insights = {
                "key_findings": answers[num_charts] if num_charts < len(answers) else "",
                "trends": answers[num_charts + 1] if num_charts + 1 < len(answers) else "",
                "recommendations": answers[num_charts + 2] if num_charts + 2 < len(answers) else "",
                "risks_opportunities": answers[num_charts + 3] if num_charts + 3 < len(answers) else ""
            }
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "dashboard_title": dashboard_config.get('title', 'Analytics Dashboard'),
                "chart_explanations": chart_explanations,
                "overall_insights": overall_insights,
                "data_sources": {
                    "company_profile": "Uploaded file",
                    "dataset": "Loaded dataset schema",
                    "charts": f"{len(charts)} generated chart(s)"
                }
            }
            
        except Exception as e:
            print(f"❌ Error generating explanations: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Error generating explanations: {str(e)}"
            }
    
    def _explain_charts_only(self, dashboard_config: Dict[str, Any], charts: List[Dict]) -> Dict[str, Any]:
        """
        Generate explanations based solely on chart analysis (no RAG/documents)
        
        Args:
            dashboard_config: Dashboard configuration
            charts: List of chart configurations
            
        Returns:
            Dictionary with chart explanations and insights
        """
        print("📊 Analyzing charts without document context...")
        
        chart_explanations = []
        
        for i, chart in enumerate(charts):
            query = chart.get('query', f'Chart {i+1}')
            chart_type = self._infer_chart_type(chart)
            chart_data = chart.get('chart_data')  # Get actual chart data if available
            
            # Parse measure and dimension from query
            measure, dimension = self._parse_query_components(query)
            
            # Build chart-focused explanation with actual data
            full_explanation = self._generate_chart_only_explanation(
                query, chart_type, measure, dimension, chart_data
            )
            
            explanation = {
                "chart_id": i + 1,
                "chart_title": query,
                "chart_type": chart_type,
                "chart_type_display": chart_type.title() + " Chart" if chart_type != 'unknown' else "Visualization",
                "x_axis": dimension if dimension else "Category",
                "y_axis": measure if measure else "Value",
                "query": query,
                "explanation": full_explanation,
                "insights": self._extract_chart_insights(query, chart_type, measure, dimension, chart_data)
            }
            chart_explanations.append(explanation)
        
        # Generate overall insights from chart patterns
        overall_insights = self._generate_overall_chart_insights(charts)
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "dashboard_title": dashboard_config.get('title', 'Analytics Dashboard'),
            "chart_explanations": chart_explanations,
            "overall_insights": overall_insights,
            "data_sources": {
                "company_profile": "Not provided",
                "dataset": "Chart data only",
                "charts": f"{len(charts)} generated chart(s)"
            },
            "mode": "chart_only"
        }
    
    def _generate_chart_only_explanation(self, query: str, chart_type: str, 
                                          measure: str, dimension: str, chart_data: Dict[str, Any] = None) -> str:
        """
        Generate intelligent AI-powered explanation for a chart
        Uses LLM to analyze the chart query and actual data values
        """
        if not LLM_AVAILABLE or not chart_llm:
            raise Exception("LLM is required for chart explanations. Please configure OPENROUTER_API_KEY.")
        
        # Prepare data summary if available
        data_context = ""
        if chart_data and isinstance(chart_data, dict):
            x_values = chart_data.get('x', [])
            y_values = chart_data.get('y', [])
            
            if x_values and y_values:
                # Create data preview
                data_context = f"\n\nACTUAL CHART DATA:\n"
                data_context += f"Categories ({dimension or 'X'}): {x_values}\n"
                data_context += f"Values ({measure or 'Y'}): {y_values}\n"
                
                # Add statistical insights
                if y_values:
                    try:
                        y_nums = [float(y) for y in y_values if y is not None]
                        if y_nums:
                            data_context += f"\nData Statistics:\n"
                            data_context += f"- Minimum: {min(y_nums):,.0f}\n"
                            data_context += f"- Maximum: {max(y_nums):,.0f}\n"
                            data_context += f"- Total Data Points: {len(y_nums)}\n"
                            
                            # Trend analysis
                            if len(y_nums) >= 2:
                                start_val = y_nums[0]
                                end_val = y_nums[-1]
                                change_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                                data_context += f"- Change from Start to End: {change_pct:+.1f}%\n"
                    except (ValueError, TypeError):
                        pass
        
        # Create prompt for LLM
        prompt_template = """You are a data visualization expert. Analyze this chart with ACTUAL DATA and provide accurate insights.

Chart Query: {query}
Chart Type: {chart_type}
Measure (Y-axis): {measure}
Dimension (X-axis): {dimension}{data_context}

CRITICAL: Base your analysis on the ACTUAL DATA PROVIDED above. Do NOT make assumptions or generic statements.

Generate a structured explanation using HTML formatting for web display:

<h3>Chart Type & Purpose</h3>
<p>[2-3 sentences about this specific chart and why it's effective for THIS data]</p>

<h3>Axes & Relationship</h3>
<p><strong>Y-Axis (Vertical):</strong> {measure} - [What this specific metric represents]</p>
<p><strong>X-Axis (Horizontal):</strong> {dimension} - [What this dimension represents]</p>
<p><strong>Relationship:</strong> [Describe the ACTUAL relationship shown in the data - trends, peaks, declines]</p>

<h3>How to Interpret</h3>
<ul>
<li>[How to read THIS specific chart with THIS data]</li>
<li>[What the ACTUAL values show - reference specific data points]</li>
<li>[How to identify the REAL trends in this data]</li>
<li>[What the changes/comparisons mean based on ACTUAL numbers]</li>
</ul>

<h3>Key Insights to Look For</h3>
<ul>
<li><strong>Trends:</strong> [Describe ACTUAL trends from the data - be specific about increases, decreases, peaks]</li>
<li><strong>Outliers:</strong> [Identify any ACTUAL unusual values from the data]</li>
<li><strong>Comparisons:</strong> [Compare ACTUAL data points - highest, lowest, changes]</li>
<li><strong>Patterns:</strong> [Describe REAL patterns visible in the data]</li>
</ul>

<h3>Business Value</h3>
<p>[2-3 sentences about what THIS SPECIFIC data tells stakeholders and what actions they should consider based on the ACTUAL trends]</p>

IMPORTANT: Use ONLY HTML tags. NO markdown. Reference ACTUAL data values and trends. Be truthful about what the data shows."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "chart_type", "measure", "dimension", "data_context"]
        )
        
        chain = prompt | chart_llm | StrOutputParser()
        
        explanation = chain.invoke({
            "query": query,
            "chart_type": chart_type or "visualization",
            "measure": measure or "values",
            "dimension": dimension or "categories",
            "data_context": data_context
        })
        
        return explanation.strip()
    
    def _extract_chart_insights(self, query: str, chart_type: str, 
                                measure: str, dimension: str, chart_data: Dict[str, Any] = None) -> List[str]:
        """
        Extract actionable insights using LLM with actual data
        """
        if not LLM_AVAILABLE or not chart_llm:
            raise Exception("LLM is required for insight extraction. Please configure OPENROUTER_API_KEY.")
        
        # Prepare data summary
        data_summary = ""
        if chart_data and isinstance(chart_data, dict):
            x_values = chart_data.get('x', [])
            y_values = chart_data.get('y', [])
            if x_values and y_values:
                data_summary = f"\nActual Data: {list(zip(x_values, y_values))}\n"
        
        prompt_template = """Based on this chart with ACTUAL DATA, provide 3-4 specific, actionable insights:

Chart Query: {query}
Measure: {measure}
Dimension: {dimension}
Chart Type: {chart_type}{data_summary}

CRITICAL: Use the ACTUAL DATA above. Reference specific values, trends, and patterns you see.

Generate concise insights that are:
- Based on REAL data values (mention specific numbers or trends)
- Actionable (what stakeholders should analyze or do)
- Business-focused
- One clear sentence each

Format as bullet points without numbering. Keep each insight to 1-2 lines maximum."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "measure", "dimension", "chart_type", "data_summary"]
        )
        
        chain = prompt | chart_llm | StrOutputParser()
        
        result = chain.invoke({
            "query": query,
            "measure": measure or "values",
            "dimension": dimension or "categories",
            "chart_type": chart_type or "visualization",
            "data_summary": data_summary
        })
        
        # Parse insights from response
        insights = [line.strip().lstrip('•-* ') for line in result.split('\n') if line.strip()]
        return insights[:4]  # Return max 4 insights
    
    def _generate_overall_chart_insights(self, charts: List[Dict]) -> Dict[str, str]:
        """
        Generate overall insights using LLM based on chart collection
        """
        if not LLM_AVAILABLE or not chart_llm:
            raise Exception("LLM is required for overall insights. Please configure OPENROUTER_API_KEY.")
        
        # Create summary of all charts WITH ACTUAL DATA
        charts_summary = []
        for i, chart in enumerate(charts, 1):
            query = chart.get('query', f'Chart {i}')
            chart_data = chart.get('chart_data', {})
            
            # Extract actual data if available
            data_info = ""
            if chart_data and isinstance(chart_data, dict):
                x_values = chart_data.get('x', [])
                y_values = chart_data.get('y', [])
                
                if x_values and y_values:
                    # Convert to string representation
                    x_str = ', '.join([str(x) for x in x_values[:10]])  # First 10 values
                    y_str = ', '.join([str(y) for y in y_values[:10]])
                    
                    # Calculate statistics - filter out None values
                    if y_values:
                        # Filter out None values for numeric calculations
                        numeric_y_values = [y for y in y_values if y is not None and isinstance(y, (int, float))]
                        
                        if numeric_y_values:
                            min_val = min(numeric_y_values)
                            max_val = max(numeric_y_values)
                            avg_val = sum(numeric_y_values) / len(numeric_y_values)
                            
                            # Calculate trend
                            if len(numeric_y_values) >= 2:
                                first_val = numeric_y_values[0]
                                last_val = numeric_y_values[-1]
                                change_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                                
                                data_info = f"\n   Data: {x_str}\n   Values: {y_str}\n   Range: {min_val:,.0f} to {max_val:,.0f}, Avg: {avg_val:,.0f}\n   Overall Change: {change_pct:+.1f}%"
            
            chart_summary = f"{i}. {query}{data_info}"
            charts_summary.append(chart_summary)
        
        charts_text = "\n".join(charts_summary)
        
        prompt_template = """Analyze this dashboard with {num_charts} chart(s) and provide well-formatted insights based on the ACTUAL DATA provided:

Charts in this dashboard WITH REAL DATA:
{charts_text}

CRITICAL INSTRUCTIONS:
- Use the ACTUAL DATA VALUES, trends, and statistics provided above
- Reference SPECIFIC NUMBERS (e.g., "peaked at 433k in FY 2024-25")
- Identify REAL PATTERNS (if data shows decline, say decline - not "steady increase")
- Base ALL insights on the data shown, NOT generic assumptions
- If Overall Change is negative, acknowledge the decline
- If values show a peak then drop, mention that pattern specifically

Generate insights using HTML formatting for web display:

<strong>KEY FINDINGS</strong>
<p>[2-3 clear sentences summarizing what this dashboard reveals based on ACTUAL data trends and values shown]</p>

<strong>TRENDS</strong>
<ul>
<li>[Trend 1: Specific pattern from the REAL data - mention actual values or percentages]</li>
<li>[Trend 2: Another pattern visible in the data - reference specific numbers]</li>
<li>[Trend 3: Additional data-driven insight with concrete values]</li>
</ul>

<strong>RECOMMENDATIONS</strong>
<ul>
<li>[Action 1: Specific recommendation based on the ACTUAL trends observed]</li>
<li>[Action 2: Another actionable step addressing real data patterns]</li>
<li>[Action 3: Strategic action based on what the data actually shows]</li>
</ul>

<strong>RISKS & OPPORTUNITIES</strong>
<ul>
<li><strong>Opportunities:</strong> [Based on positive trends in real data]</li>
<li><strong>Risks:</strong> [Based on concerning patterns in actual data]</li>
<li><strong>Focus Areas:</strong> [Where to concentrate efforts based on data insights]</li>
</ul>

IMPORTANT: 
- Use ONLY HTML tags (strong, p, ul, li). NO markdown symbols like **, *, or ##.
- Reference SPECIFIC VALUES from the data provided
- If data shows decline/drop, say so - don't claim "steady increase"
- Use concise, business-focused language. Each point should be 1-2 lines max."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["num_charts", "charts_text"]
        )
        
        chain = prompt | chart_llm | StrOutputParser()
        
        result = chain.invoke({
            "num_charts": len(charts),
            "charts_text": charts_text
        })
        
        # Parse the response
        sections = {
            "key_findings": "",
            "trends": "",
            "recommendations": "",
            "risks_opportunities": ""
        }
        
        current_section = None
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if 'KEY FINDINGS' in line.upper():
                current_section = 'key_findings'
            elif 'TRENDS' in line.upper():
                current_section = 'trends'
            elif 'RECOMMENDATIONS' in line.upper():
                current_section = 'recommendations'
            elif 'RISKS' in line.upper() or 'OPPORTUNITIES' in line.upper():
                current_section = 'risks_opportunities'
            elif current_section:
                sections[current_section] += line + "\n"
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def _create_chart_documents(self, charts: List[Dict]) -> List[Document]:
        """
        Create document chunks from chart information for RAG context
        """
        chart_docs = []
        
        for i, chart in enumerate(charts):
            query = chart.get('query', '')
            timestamp = chart.get('timestamp', '')
            
            # Format chart information as text
            chart_text = f"CHART {i+1}:\n"
            chart_text += f"Query: {query}\n"
            if timestamp:
                chart_text += f"Created: {timestamp}\n"
            
            # Infer what the chart shows from the query
            chart_text += self._describe_chart_from_query(query)
            
            doc = Document(
                page_content=chart_text,
                metadata={
                    "source": "generated_chart",
                    "type": "chart_info",
                    "chart_id": i + 1
                }
            )
            chart_docs.append(doc)
        
        return chart_docs
    
    def _describe_chart_from_query(self, query: str) -> str:
        """
        Extract meaningful description from query with detailed visualization explanation
        """
        description = "\nVisualization Details:\n"
        query_lower = query.lower()
        
        # Determine chart type
        chart_type = "bar chart"
        if 'line' in query_lower:
            chart_type = "line chart"
        elif 'pie' in query_lower:
            chart_type = "pie chart"
        elif 'scatter' in query_lower:
            chart_type = "scatter plot"
        
        description += f"Chart Type: {chart_type.title()}\n"
        
        # Extract measures and dimensions from query (what's being shown)
        measure = ""
        dimension = ""
        
        if ' by ' in query_lower:
            parts = query.split(' by ', 1)
            if len(parts) >= 2:
                measure = parts[0].strip()
                dimension = parts[1].strip()
                
                # Clean up common prefixes
                measure = measure.replace('show ', '').replace('display ', '').strip()
                
                description += f"Measure (Y-axis): {measure}\n"
                description += f"Dimension (X-axis): {dimension}\n"
                description += f"\nWhat it shows: This {chart_type} displays {measure} across different {dimension}. "
                
                # Add interpretation based on chart type
                if 'bar' in chart_type:
                    description += f"Each bar represents a distinct {dimension}, with the height showing the {measure}. "
                    description += "This allows easy comparison between categories.\n"
                elif 'line' in chart_type:
                    description += f"The line connects data points across {dimension}, showing trends and patterns in {measure} over time or categories. "
                    description += "This helps identify trends, seasonality, and changes.\n"
                elif 'pie' in chart_type:
                    description += f"Each slice represents the proportion of {measure} for each {dimension}. "
                    description += "This shows the relative distribution and composition of the total.\n"
        
        # Check for aggregations
        aggregation = ""
        if 'sum' in query_lower or 'total' in query_lower:
            aggregation = "sum (total)"
            description += f"Aggregation: Total {measure} is calculated by summing all values. "
        elif 'avg' in query_lower or 'average' in query_lower:
            aggregation = "average (mean)"
            description += f"Aggregation: Average {measure} is calculated as the mean of all values. "
        elif 'count' in query_lower:
            aggregation = "count"
            description += f"Aggregation: Count of records is displayed. "
        
        # Add insights guidance
        description += "\nPotential Insights:\n"
        description += "- Compare values across categories to identify highest and lowest performers\n"
        description += "- Look for patterns, trends, or anomalies in the data\n"
        description += "- Identify outliers or unusual values that need investigation\n"
        description += "- Use these insights to make data-driven business decisions\n"
        
        # Check for filters
        if 'top' in query_lower:
            import re
            top_match = re.search(r'top\s+(\d+)', query_lower)
            if top_match:
                n = top_match.group(1)
                description += f"\nFilter Applied: Showing only top {n} records based on the measure value.\n"
        
        return description
    
    def _infer_chart_type(self, chart: Dict) -> str:
        """
        Infer chart type from query
        """
        query = chart.get('query', '').lower()
        
        if 'bar' in query:
            return 'bar'
        elif 'line' in query:
            return 'line'
        elif 'pie' in query:
            return 'pie'
        elif 'scatter' in query:
            return 'scatter'
        else:
            return 'unknown'
    
    def _generate_chart_questions(self, charts: List[Dict]) -> List[str]:
        """
        Generate relevant questions for each chart based on its type and content
        """
        questions = []
        
        for i, chart in enumerate(charts):
            query = chart.get('query', f'Chart {i+1}')
            
            # Parse the query to extract measure and dimension
            measure, dimension = self._parse_query_components(query)
            
            # Infer chart type
            chart_type = self._infer_chart_type(chart)
            chart_type_name = "visualization"
            if chart_type == 'bar':
                chart_type_name = "bar chart"
            elif chart_type == 'line':
                chart_type_name = "line chart"
            elif chart_type == 'pie':
                chart_type_name = "pie chart"
            elif chart_type == 'scatter':
                chart_type_name = "scatter plot"
            
            # Generate detailed contextual question
            question = f"Chart {i+1}: '{query}'\n\n"
            question += f"This {chart_type_name} displays "
            
            if measure and dimension:
                question += f"{measure} (Y-axis) across different {dimension} (X-axis). "
            else:
                question += f"the visualization of {query}. "
            
            question += "\n\nBased on the uploaded company profile and dataset information, please explain:\n"
            question += f"1. What type of chart is this ({chart_type_name})?\n"
            question += f"2. What does the X-axis represent ({dimension if dimension else 'category'})?\n"
            question += f"3. What does the Y-axis represent ({measure if measure else 'value'})?\n"
            question += "4. How do the X-axis and Y-axis relate to each other?\n"
            question += "5. What business insights can users derive from this visualization?\n"
            question += "6. What trends, patterns, or key findings are visible?\n"
            question += "7. What actions or decisions should be made based on this data?\n"
            
            questions.append(question)
        
        return questions
    
    def _extract_insights(self, explanation: str) -> List[str]:
        """
        Extract key insights from explanation text
        """
        if not explanation:
            return []
        
        # Simple insight extraction - split by sentences and filter meaningful ones
        sentences = explanation.split('.')
        insights = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30 and any(keyword in sentence.lower() for keyword in 
                ['increase', 'decrease', 'trend', 'highest', 'lowest', 'significant', 
                 'important', 'shows', 'indicates', 'suggests', 'reveals']):
                insights.append(sentence + '.')
        
        return insights[:3]  # Return top 3 insights
    
    def _parse_query_components(self, query: str) -> tuple:
        """
        Safely parse measure and dimension from query (case-insensitive)
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (measure, dimension)
        """
        import re
        
        measure = ""
        dimension = ""
        
        # Case-insensitive search for ' by '
        match = re.search(r'\s+by\s+', query, re.IGNORECASE)
        if match:
            # Split at the matched position
            split_pos = match.start()
            measure = query[:split_pos].replace('Show ', '').replace('show ', '').strip()
            dimension = query[split_pos + len(match.group()):].strip()
        
        return measure, dimension
    
    def explain_single_chart(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for a single chart
        """
        try:
            # Check if documents are available
            has_documents = bool(self.company_docs or self.dataset_docs)
            
            if not has_documents:
                # No documents - use chart-only explanation
                print("📊 Generating chart-only explanation (no documents)...")
                query = chart_config.get('query', 'Chart')
                chart_type = self._infer_chart_type(chart_config)
                
                measure, dimension = self._parse_query_components(query)
                
                explanation = self._generate_chart_only_explanation(
                    query, chart_type, measure, dimension
                )
                
                return {
                    "success": True,
                    "chart_title": chart_config.get('title', 'Chart'),
                    "explanation": explanation,
                    "insights": self._extract_chart_insights(query, chart_type, measure, dimension),
                    "mode": "chart_only"
                }
            
            # Documents available - use RAG
            if not self.vectorstore:
                return {
                    "success": False,
                    "error": "RAG system not initialized"
                }
            
            print("🔍 Generating RAG-enhanced explanation...")
            questions = self._generate_chart_questions([chart_config])
            answers = process_questions_with_rag(self.vectorstore,
                                                self.company_docs + self.dataset_docs if self.company_docs and self.dataset_docs else [],
                                                questions)
            
            return {
                "success": True,
                "chart_title": chart_config.get('title', 'Chart'),
                "explanation": answers[0] if answers else "No explanation available",
                "insights": self._extract_insights(answers[0] if answers else ""),
                "mode": "rag_enhanced"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error explaining chart: {str(e)}"
            }
    
    def get_comparative_insights(self, charts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparative insights across multiple charts
        """
        try:
            # Check if documents are available
            has_documents = bool(self.company_docs or self.dataset_docs)
            
            if not has_documents:
                # No documents - generate chart-based comparative insights
                print("📊 Generating comparative insights from charts only...")
                overall_insights = self._generate_overall_chart_insights(charts)
                
                return {
                    "success": True,
                    "comparative_analysis": overall_insights.get("key_findings", ""),
                    "overall_performance": overall_insights.get("trends", ""),
                    "patterns": overall_insights.get("risks_opportunities", ""),
                    "complete_story": overall_insights.get("recommendations", ""),
                    "mode": "chart_only"
                }
            
            # Documents available - use RAG
            if not self.vectorstore:
                return {
                    "success": False,
                    "error": "RAG system not initialized"
                }
            
            print("🔍 Generating RAG-enhanced comparative insights...")
            # Generate comparative questions
            questions = [
                "Compare the key metrics across all visualizations. What relationships exist between them?",
                "What is the overall business performance based on all the charts?",
                "Are there any contradictions or surprising patterns across the different visualizations?",
                "What is the complete story told by this dashboard when all charts are considered together?"
            ]
            
            answers = process_questions_with_rag(self.vectorstore,
                                                self.company_docs + self.dataset_docs if self.company_docs and self.dataset_docs else [],
                                                questions)
            
            return {
                "success": True,
                "comparative_analysis": answers[0] if len(answers) > 0 else "",
                "overall_performance": answers[1] if len(answers) > 1 else "",
                "patterns": answers[2] if len(answers) > 2 else "",
                "complete_story": answers[3] if len(answers) > 3 else "",
                "mode": "rag_enhanced"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating comparative insights: {str(e)}"
            }

# Global explainer instance
_explainer_instance = None

def get_dashboard_explainer() -> DashboardExplainer:
    """Get or create global dashboard explainer instance"""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = DashboardExplainer()
    return _explainer_instance
