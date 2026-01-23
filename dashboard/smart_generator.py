"""
Smart Dashboard Generator - AI-powered automatic chart recommendations
Provides context-aware dashboard generation without requiring user queries.

Architecture:
1. DataProfiler: Analyzes dataset schema and statistics
2. ContextAnalyzer: Extracts user/organizational context
3. SmartChartRecommender: LLM-powered chart recommendations
4. SmartDashboardGenerator: Orchestrates the full workflow

Usage:
    generator = SmartDashboardGenerator(data_connector, use_llm=True)
    recommendations = generator.generate_smart_dashboard(
        user_department="Finance",
        user_role="Analyst", 
        override_context=None
    )
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from charts.data_connector import DataConnector
from charts.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ChartRecommendation(BaseModel):
    """Single chart recommendation from LLM"""
    metric: str = Field(description="Numeric column to measure")
    dimension: str = Field(description="Categorical column to group by")
    chart_type: str = Field(description="Chart type: bar, line, pie, area, scatter, heatmap, radar")
    aggregation: str = Field(description="Aggregation: sum, avg, count, min, max")
    title: str = Field(description="Human-readable chart title")
    reasoning: str = Field(description="Why this chart is recommended")
    priority: int = Field(description="Priority ranking (1=highest)")
    filters: List[str] = Field(default_factory=list, description="Optional filter conditions")
    limit: Optional[int] = Field(default=None, description="Result limit (e.g., top 10)")
    calculation_type: Optional[str] = Field(default=None, description="Calculated metric type")
    group_by: Optional[str] = Field(default=None, description="Secondary grouping dimension")
    time_granularity: Optional[str] = Field(default=None, description="Time aggregation level")


class RecommendationList(BaseModel):
    """List of chart recommendations"""
    recommendations: List[ChartRecommendation] = Field(description="List of 5 chart recommendations")


@dataclass
class DataProfile:
    """Dataset profile with metadata for LLM analysis"""
    dataset_name: str
    tables: Dict[str, Dict[str, Any]]  # {table_name: {numeric: [], text: [], date: [], row_count: int}}
    total_rows: int
    total_columns: int
    sample_data: Optional[Dict[str, pd.DataFrame]] = None  # Optional sample data for context
    industry: Optional[str] = None  # Detected industry (retail, saas, manufacturing, healthcare, finance)
    semantic_columns: Optional[Dict[str, List[str]]] = None  # {location: [], time: [], segment: [], product: []}
    relationships: Optional[List[str]] = None  # Detected data relationships (e.g., "Branch + Year → trends")


@dataclass
class UserContext:
    """User context for personalized recommendations"""
    department: Optional[str]
    role: str
    preferred_metrics: List[str]  # Based on department
    preferred_chart_types: List[str]
    avoid_chart_types: List[str]


# ============================================================================
# Component 1: DataProfiler
# ============================================================================

class DataProfiler:
    """
    Analyzes dataset characteristics to provide LLM with structured context
    
    Capabilities:
    - Detect column types (numeric, categorical, temporal)
    - Calculate basic statistics (cardinality, null rates)
    - Identify potential relationships
    - Sample representative data
    - Industry detection
    - Semantic column categorization (location, time, segment, product)
    """
    
    # Semantic column patterns (handles different naming conventions)
    SEMANTIC_PATTERNS = {
        'location': ['branch', 'location', 'store', 'office', 'region', 'state', 'country', 'city', 'area', 'territory', 'zone', 'market', 'geography'],
        'time': ['year', 'month', 'quarter', 'date', 'day', 'week', 'period', 'fiscal', 'time', 'timestamp'],
        'segment': ['category', 'segment', 'type', 'class', 'group', 'division', 'department', 'unit'],
        'product': ['product', 'item', 'sku', 'service', 'offering', 'goods', 'merchandise'],
        'customer': ['customer', 'client', 'account', 'buyer', 'user', 'member'],
        'financial': ['revenue', 'sales', 'cost', 'profit', 'margin', 'price', 'value', 'amount', 'total', 'rate', 'expense']
    }
    
    # Industry detection keywords
    INDUSTRY_KEYWORDS = {
        'retail': ['store', 'branch', 'inventory', 'goods', 'merchandise', 'insulated', 'non-insulated', 'retail', 'product', 'sku'],
        'saas': ['subscription', 'mrr', 'arr', 'churn', 'plan', 'tier', 'license', 'user'],
        'manufacturing': ['production', 'units', 'defects', 'capacity', 'throughput', 'yield', 'assembly'],
        'healthcare': ['patient', 'admission', 'diagnosis', 'treatment', 'claims', 'provider'],
        'finance': ['transaction', 'portfolio', 'assets', 'liabilities', 'investment', 'loan'],
        'ecommerce': ['order', 'cart', 'checkout', 'shipping', 'delivery', 'online']
    }
    
    def __init__(self, data_connector: DataConnector):
        self.data_connector = data_connector
    
    def profile_current_dataset(self, include_sample: bool = False) -> DataProfile:
        """
        Profile the currently active dataset
        
        Args:
            include_sample: Whether to include sample data (increases context size)
        
        Returns:
            DataProfile with schema and statistics
        """
        try:
            dataset_name = self.data_connector.get_current_dataset()
            if not dataset_name:
                raise ValueError("No active dataset loaded")
            
            # Extract column information from DataConnector
            column_info = self.data_connector.extract_all_columns_info()
            
            total_rows = 0
            total_columns = 0
            sample_data = {} if include_sample else None
            
            # Process each table
            tables = {}
            for table_name, info in column_info.items():
                tables[table_name] = {
                    'numeric': info.get('numeric_columns', []),
                    'text': info.get('text_columns', []),
                    'date': info.get('date_columns', []),
                    'row_count': info.get('row_count', 0)
                }
                
                total_rows += info.get('row_count', 0)
                total_columns += len(info.get('numeric_columns', [])) + \
                                len(info.get('text_columns', [])) + \
                                len(info.get('date_columns', []))
                
                # Optional: Get sample data (first 3 rows)
                if include_sample and table_name in self.data_connector.cached_data:
                    df = self.data_connector.cached_data[table_name]
                    sample_data[table_name] = df.head(3)
            
            # Detect industry
            industry = self._detect_industry(tables)
            
            # Detect semantic column types
            semantic_columns = self._detect_semantic_columns(tables)
            
            # Detect data relationships
            relationships = self._detect_relationships(tables, semantic_columns)
            
            profile = DataProfile(
                dataset_name=dataset_name,
                tables=tables,
                total_rows=total_rows,
                total_columns=total_columns,
                sample_data=sample_data,
                industry=industry,
                semantic_columns=semantic_columns,
                relationships=relationships
            )
            
            logger.info(f"📊 Data Profile: {dataset_name} - {total_rows} rows, {total_columns} columns")
            logger.info(f"🏭 Detected Industry: {industry}")
            logger.info(f"🔗 Detected Relationships: {len(relationships)} patterns")
            return profile
            
        except Exception as e:
            logger.error(f"❌ Error profiling dataset: {e}")
            raise
    
    def get_recommended_metrics(self, profile: DataProfile) -> List[str]:
        """
        Suggest which metrics are most interesting for visualization
        
        Strategy:
        - Prefer numeric columns with high variance
        - Exclude ID-like columns (e.g., customer_id, order_id)
        """
        recommended = []
        
        for table_name, info in profile.tables.items():
            numeric_cols = info['numeric']
            
            # Simple heuristic: exclude columns with 'id' in name
            for col in numeric_cols:
                col_lower = col.lower()
                if 'id' not in col_lower and 'key' not in col_lower:
                    recommended.append(col)
        
        return recommended[:10]  # Top 10 most promising metrics
    
    def get_recommended_dimensions(self, profile: DataProfile) -> List[str]:
        """
        Suggest which dimensions are most interesting for grouping
        
        Strategy:
        - Prefer categorical columns with moderate cardinality (2-50 values)
        - Include date columns for time-series analysis
        """
        recommended = []
        
        for table_name, info in profile.tables.items():
            # Add text/categorical columns
            text_cols = info['text']
            for col in text_cols:
                col_lower = col.lower()
                if 'id' not in col_lower and 'key' not in col_lower and 'description' not in col_lower:
                    recommended.append(col)
            
            # Add date columns (high priority for time-series)
            date_cols = info['date']
            recommended.extend(date_cols)
        
        return recommended[:10]
    
    def _detect_industry(self, tables: Dict[str, Dict[str, Any]]) -> str:
        """
        Detect industry based on column names and patterns
        
        Returns:
            Industry name or 'general'
        """
        all_columns = []
        for table_info in tables.values():
            all_columns.extend(table_info.get('numeric', []))
            all_columns.extend(table_info.get('text', []))
            all_columns.extend(table_info.get('date', []))
        
        # Convert to lowercase for matching
        columns_lower = ' '.join([col.lower() for col in all_columns])
        
        # Score each industry
        industry_scores = {}
        for industry, keywords in self.INDUSTRY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in columns_lower)
            if score > 0:
                industry_scores[industry] = score
        
        # Return highest scoring industry
        if industry_scores:
            detected = max(industry_scores, key=industry_scores.get)
            logger.info(f"🏭 Industry detected: {detected} (score: {industry_scores[detected]})")
            return detected
        
        return 'general'
    
    def _detect_semantic_columns(self, tables: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Categorize columns into semantic types (location, time, segment, product, etc.)
        Handles different naming conventions (Branch vs Location vs Store)
        
        Returns:
            Dict of semantic categories with matching columns
        """
        semantic_map = {
            'location': [],
            'time': [],
            'segment': [],
            'product': [],
            'customer': [],
            'financial': []
        }
        
        for table_info in tables.values():
            all_cols = table_info.get('numeric', []) + table_info.get('text', []) + table_info.get('date', [])
            
            for col in all_cols:
                col_lower = col.lower()
                
                # Match against semantic patterns
                for semantic_type, patterns in self.SEMANTIC_PATTERNS.items():
                    if any(pattern in col_lower for pattern in patterns):
                        if col not in semantic_map[semantic_type]:  # Avoid duplicates
                            semantic_map[semantic_type].append(col)
        
        # Log detected semantic columns
        for sem_type, cols in semantic_map.items():
            if cols:
                logger.info(f"🏷️  {sem_type.capitalize()}: {cols}")
        
        return semantic_map
    
    def _detect_relationships(self, tables: Dict[str, Dict[str, Any]], semantic_columns: Dict[str, List[str]]) -> List[str]:
        """
        Detect meaningful data relationships for chart recommendations
        
        Patterns:
        - Location + Time → Trend analysis by location
        - Segment + Financial → Category performance
        - Product + Time → Product lifecycle
        - Location + Segment → Cross-dimensional analysis
        
        Returns:
            List of relationship descriptions
        """
        relationships = []
        
        locations = semantic_columns.get('location', [])
        times = semantic_columns.get('time', [])
        segments = semantic_columns.get('segment', [])
        products = semantic_columns.get('product', [])
        financials = semantic_columns.get('financial', [])
        
        # Location + Time relationships
        if locations and times:
            for loc in locations:
                for time in times:
                    relationships.append(f"{loc} + {time} → Multi-series time trends (e.g., branch performance over years)")
        
        # Segment + Financial relationships
        if segments and financials:
            for seg in segments:
                for fin in financials:
                    relationships.append(f"{seg} + {fin} → Category performance analysis (e.g., revenue by product category)")
        
        # Product + Time relationships
        if products and times:
            for prod in products:
                for time in times:
                    relationships.append(f"{prod} + {time} → Product lifecycle trends (e.g., product sales over time)")
        
        # Location + Segment relationships
        if locations and segments:
            for loc in locations:
                for seg in segments:
                    relationships.append(f"{loc} + {seg} → Cross-dimensional analysis (e.g., category dominance by branch)")
        
        # Time + Financial relationships (always valuable)
        if times and financials:
            for time in times:
                for fin in financials:
                    relationships.append(f"{time} + {fin} → Growth analysis (e.g., year-on-year revenue growth)")
        
        return relationships[:15]  # Limit to top 15 most relevant


# ============================================================================
# Component 2: ContextAnalyzer
# ============================================================================

class ContextAnalyzer:
    """
    Extracts user/organizational context for personalized recommendations
    
    Uses department and role to suggest relevant metrics and chart types
    """
    
    # Department-specific metric preferences
    DEPARTMENT_METRICS = {
        'finance': ['revenue', 'sales', 'cost', 'profit', 'margin', 'expenses', 'budget', 'variance', 'amount', 'value', 'total'],
        'marketing': ['conversions', 'leads', 'engagement', 'roi', 'clicks', 'impressions', 'spend', 'sales', 'revenue', 'quantity'],
        'sales': ['deals', 'pipeline', 'win_rate', 'quota', 'bookings', 'closed', 'opportunities', 'revenue', 'sales', 'quantity', 'amount'],
        'operations': ['throughput', 'efficiency', 'capacity', 'utilization', 'defects', 'cycle_time', 'quantity', 'volume', 'orders'],
        'hr': ['headcount', 'retention', 'turnover', 'satisfaction', 'productivity', 'hiring'],
        'customer_success': ['churn', 'nps', 'csat', 'tickets', 'resolution_time', 'satisfaction'],
        'general': ['sales', 'revenue', 'quantity', 'amount', 'total', 'value', 'count', 'volume'],  # Default for no department
    }
    
    # Role-specific chart type preferences
    ROLE_PREFERENCES = {
        'Admin': {
            'preferred': ['bar', 'line', 'area', 'heatmap'],
            'avoid': []  # Admins see everything
        },
        'Analyst': {
            'preferred': ['line', 'scatter', 'heatmap', 'area'],
            'avoid': ['pie']  # Analysts prefer precise visualizations
        },
        'Departmental': {
            'preferred': ['bar', 'line', 'pie'],
            'avoid': ['scatter', 'heatmap']  # Simpler charts
        },
        'Viewer': {
            'preferred': ['bar', 'pie', 'line'],
            'avoid': ['scatter', 'heatmap', 'radar']  # Very simple
        }
    }
    
    def __init__(self):
        pass
    
    def analyze_context(
        self, 
        user_department: Optional[str],
        user_role: str,
        override_context: Optional[Dict[str, Any]] = None
    ) -> UserContext:
        """
        Build user context for recommendations
        
        Args:
            user_department: User's department (from auth)
            user_role: User's role (Admin, Analyst, Departmental, Viewer)
            override_context: Optional override (e.g., {"department": "marketing"})
        
        Returns:
            UserContext with preferences
        """
        # Apply override if provided
        if override_context:
            user_department = override_context.get('department', user_department)
            user_role = override_context.get('role', user_role)
        
        # Get department-specific metrics
        dept_key = (user_department or '').lower().replace(' ', '_')
        preferred_metrics = self.DEPARTMENT_METRICS.get(dept_key, [])
        
        # Get role-specific chart preferences
        role_prefs = self.ROLE_PREFERENCES.get(user_role, self.ROLE_PREFERENCES['Viewer'])
        
        context = UserContext(
            department=user_department,
            role=user_role,
            preferred_metrics=preferred_metrics,
            preferred_chart_types=role_prefs['preferred'],
            avoid_chart_types=role_prefs['avoid']
        )
        
        logger.info(f"👤 User Context: {user_role} in {user_department} - {len(preferred_metrics)} priority metrics")
        return context


# ============================================================================
# Component 3: SmartChartRecommender (LLM-Powered)
# ============================================================================

class SmartChartRecommender:
    """
    LLM-powered chart recommendation engine
    
    Uses Claude 3 Haiku (for MVP, upgrade to Sonnet later) to generate
    contextually relevant chart recommendations based on data + user context
    """
    
    def __init__(self, api_key: Optional[str] = None, use_llm: bool = True):
        self.use_llm = use_llm
        
        if self.use_llm:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                logger.warning("⚠️  No OPENROUTER_API_KEY found. Using fallback recommendations.")
                self.use_llm = False
            else:
                # Initialize LLM (Claude 3 Haiku for MVP)
                self.llm = ChatOpenAI(
                    model="anthropic/claude-3-haiku",
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.3,  # Focused recommendations
                    max_tokens=1500
                )
                logger.info("✅ LLM initialized: Claude 3 Haiku")
    
    def recommend_charts(
        self,
        data_profile: DataProfile,
        user_context: UserContext,
        num_recommendations: int = 5,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate chart recommendations
        
        Args:
            data_profile: Dataset metadata
            user_context: User preferences
            num_recommendations: Number of charts to recommend
            custom_prompt: Optional custom instructions (e.g., "focus on CEO metrics", "highlight product X")
        
        Returns:
            List of chart recommendations (as entity dicts)
        """
        if self.use_llm:
            return self._recommend_with_llm(data_profile, user_context, num_recommendations, custom_prompt)
        else:
            return self._recommend_fallback(data_profile, user_context, num_recommendations)
    
    def _recommend_with_llm(
        self,
        data_profile: DataProfile,
        user_context: UserContext,
        num_recommendations: int,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """LLM-based recommendations using Claude Haiku"""
        try:
            # Build prompt with data schema and user context
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert business intelligence analyst specializing in executive-level data visualizations.

Your goal: Generate ACTIONABLE, INSIGHT-DRIVEN charts that answer critical business questions for executives (CEOs, department heads, analysts).

CRITICAL INSTRUCTIONS:
1. Only use columns that exist in the provided schema
2. Match chart types to analytical goals:
   - bar: Categorical comparisons (branch vs branch, category vs category, state vs state)
   - line: Time trends and growth analysis (year-over-year, month-over-month, multi-series for comparisons)
   - pie: Part-to-whole comparisons (revenue contribution %, market share)
   - area: Cumulative trends and stacked comparisons over time
   - scatter: Correlations between metrics (price vs quantity, rate vs sales)
   - heatmap: Cross-dimensional patterns (category performance across branches)
3. **Prioritize COMPARISON and TREND analysis** over simple aggregations
4. **Use time dimensions when available** for growth/trend charts
5. **Multi-series line/area charts** for comparing segments over time (e.g., branch A vs branch B sales over years)
6. Consider user department priorities and role
7. Ensure variety in chart types

EXECUTIVE BUSINESS QUESTIONS (Use these as templates):
- **Growth Analysis**: "What is the year-on-year growth in sales revenue?" → Line chart with YoY calculation
- **Performance Comparison**: "Which branch/location generated higher total sales across years?" → Multi-series line chart
- **Geographic Trends**: "How do sales trends differ by state/region?" → Multi-series line or bar chart
- **Product Performance**: "Which products consistently perform best in terms of revenue and quantity?" → Bar chart with top performers
- **Segment Comparison**: "Compare [Segment A] vs [Segment B] in terms of revenue, average rate, and growth trend" → Multiple charts or grouped bar
- **Dominance Analysis**: "Which product category dominates in each branch/location?" → Grouped bar or heatmap
- **Seasonal Patterns**: "Are there seasonal spikes in sales for certain categories?" → Line chart by month/quarter
- **Average Metrics**: "What is the average [metric] per order/customer across years?" → Line chart for trends
- **Impact Analysis**: "How does [factor X] affect [metric Y]?" → Scatter plot or grouped bar
- **Simple Breakdowns**: "Sales by year/category/branch/state" → Bar or pie charts

RELATIONSHIP-DRIVEN RECOMMENDATIONS:
- If you see **Location + Time columns** → Create trend comparisons (multi-series line charts showing each location's performance over time)
- If you see **Segment/Category + Financial metric** → Create performance analysis (which segment contributes most revenue?)
- If you see **Product + Time** → Create lifecycle trends (how product sales evolve over time)
- If you see **Location + Segment** → Create cross-dimensional analysis (which category performs best in each branch?)

Return EXACTLY {num_recommendations} recommendations in JSON format."""),
                ("user", """Dataset Schema:
{schema}

Industry Context: {industry}

Detected Data Relationships:
{relationships}

Semantic Column Types:
- Location/Branch columns: {location_cols}
- Time/Date columns: {time_cols}
- Segment/Category columns: {segment_cols}
- Product columns: {product_cols}
- Financial/Metric columns: {financial_cols}

User Context:
- Department: {department}
- Role: {role}
- Priority Metrics: {priority_metrics}
- Preferred Chart Types: {preferred_charts}
{custom_instructions}

Generate {num_recommendations} chart recommendations as JSON array with this structure:
[
  {{
    "metric": "column_name",
    "dimension": "column_name",
    "chart_type": "bar|line|pie|area|scatter|heatmap",
    "aggregation": "sum|avg|count|min|max",
    "title": "Executive-Friendly Descriptive Title",
    "reasoning": "Why this chart answers a key business question for {role} in {department}",
    "priority": 1,
    "group_by": "optional_second_dimension_for_multi_series",
    "calculation_type": "yoy_growth|mom_change|cumulative|percent_change"
  }}
]

REQUIREMENTS:
✅ **Use the Detected Relationships above** to guide your recommendations
✅ **Prioritize multi-dimensional analysis**: If location + time exist → create multi-series line charts comparing locations over time
✅ **Include growth/trend calculations**: For time-based charts, consider YoY growth or cumulative analysis
✅ **Create comparison charts**: Branch vs Branch, Category vs Category, State vs State
✅ **Use semantic column types**: Match location columns with aggregated metrics, time columns with trends
✅ **Ensure metric is numeric** (sales, revenue, quantity, rate, value, amount)
✅ **Ensure dimension is categorical or date** (branch, state, category, product, year, month)
✅ **Business-friendly titles**: "Year-on-Year Sales Growth by Branch" not "Sales by Year and Branch"
✅ **Top performers**: Include "Top 10 Products by Revenue" or "Top 5 Branches by Sales" with limit=10 or limit=5
✅ **Cross-dimensional insights**: "Which category dominates in each branch?" → use heatmap or grouped bar
✅ **For {department} department**: Focus on {priority_metrics} metrics
✅ **Industry-specific**: This is {industry} data - prioritize relevant metrics (e.g., retail → store/product analysis)

🎯 GOAL: Each chart should answer a specific executive question about performance, trends, comparisons, or insights.""")
            ])
            
            # Format schema for prompt
            schema_text = self._format_schema_for_prompt(data_profile)
            
            # Extract semantic columns for prompt
            semantic_cols = data_profile.semantic_columns or {}
            location_cols = ', '.join(semantic_cols.get('location', [])) or 'None'
            time_cols = ', '.join(semantic_cols.get('time', [])) or 'None'
            segment_cols = ', '.join(semantic_cols.get('segment', [])) or 'None'
            product_cols = ', '.join(semantic_cols.get('product', [])) or 'None'
            financial_cols = ', '.join(semantic_cols.get('financial', [])) or 'None'
            
            # Format relationships
            relationships_text = '\n'.join(f"  • {rel}" for rel in (data_profile.relationships or [])) or '  • No specific relationships detected'
            
            # Format custom prompt if provided
            custom_instructions = ""
            if custom_prompt:
                custom_instructions = f"\n\n⭐ SPECIAL INSTRUCTIONS: {custom_prompt}"
                logger.info(f"💬 Custom prompt: {custom_prompt}")
            
            # Build prompt
            prompt = prompt_template.format_messages(
                schema=schema_text,
                industry=data_profile.industry or 'general',
                relationships=relationships_text,
                location_cols=location_cols,
                time_cols=time_cols,
                segment_cols=segment_cols,
                product_cols=product_cols,
                financial_cols=financial_cols,
                department=user_context.department or "General",
                role=user_context.role,
                priority_metrics=', '.join(user_context.preferred_metrics) or "Any",
                preferred_charts=', '.join(user_context.preferred_chart_types),
                custom_instructions=custom_instructions,
                num_recommendations=num_recommendations
            )
            
            # Call LLM
            logger.info("🤖 Calling Claude Haiku for chart recommendations...")
            response = self.llm.invoke(prompt)
            
            # Debug: Log raw response
            logger.debug(f"📥 LLM Raw Response: {response.content[:500]}...")
            
            # Parse JSON response
            recommendations = self._parse_llm_response(response.content)
            
            # Validate and convert to entity dicts
            valid_recommendations = []
            for rec in recommendations:
                if self._validate_recommendation(rec, data_profile):
                    entity_dict = self._recommendation_to_entities(rec)
                    valid_recommendations.append(entity_dict)
            
            logger.info(f"✅ Generated {len(valid_recommendations)} valid recommendations")
            return valid_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"❌ LLM recommendation failed: {e}")
            logger.info("⚠️  Falling back to rule-based recommendations")
            return self._recommend_fallback(data_profile, user_context, num_recommendations)
    
    def _recommend_fallback(
        self,
        data_profile: DataProfile,
        user_context: UserContext,
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback rule-based recommendations when LLM unavailable
        
        Simple heuristics:
        1. Find first numeric column + first categorical column → bar chart
        2. If date column exists + numeric → line chart
        3. Top 10 by dimension → bar chart with limit
        """
        recommendations = []
        
        # Get first table (simplified for MVP)
        if not data_profile.tables:
            return []
        
        table_name = list(data_profile.tables.keys())[0]
        table_info = data_profile.tables[table_name]
        
        numeric_cols = table_info['numeric']
        text_cols = table_info['text']
        date_cols = table_info['date']
        
        # Recommendation 1: Basic bar chart
        if numeric_cols and text_cols:
            recommendations.append({
                'metric': numeric_cols[0],
                'dimension': text_cols[0],
                'chart_type': 'bar',
                'aggregation': 'sum',
                'title': f'Total {numeric_cols[0]} by {text_cols[0]}',
                'filters': [],
                'limit': None
            })
        
        # Recommendation 2: Time series if date exists
        if numeric_cols and date_cols:
            recommendations.append({
                'metric': numeric_cols[0],
                'dimension': date_cols[0],
                'chart_type': 'line',
                'aggregation': 'sum',
                'title': f'{numeric_cols[0]} Trend Over Time',
                'filters': [],
                'limit': None
            })
        
        # Recommendation 3: Top 10
        if numeric_cols and text_cols:
            recommendations.append({
                'metric': numeric_cols[0],
                'dimension': text_cols[0],
                'chart_type': 'bar',
                'aggregation': 'sum',
                'title': f'Top 10 {text_cols[0]} by {numeric_cols[0]}',
                'filters': [],
                'limit': 10
            })
        
        # Recommendation 4: Second metric if available
        if len(numeric_cols) > 1 and text_cols:
            recommendations.append({
                'metric': numeric_cols[1],
                'dimension': text_cols[0],
                'chart_type': 'bar',
                'aggregation': 'avg',
                'title': f'Average {numeric_cols[1]} by {text_cols[0]}',
                'filters': [],
                'limit': None
            })
        
        # Recommendation 5: Pie chart for distribution
        if numeric_cols and text_cols and 'pie' not in user_context.avoid_chart_types:
            recommendations.append({
                'metric': numeric_cols[0],
                'dimension': text_cols[0],
                'chart_type': 'pie',
                'aggregation': 'sum',
                'title': f'{numeric_cols[0]} Distribution by {text_cols[0]}',
                'filters': [],
                'limit': 8  # Limit pie slices
            })
        
        logger.info(f"⚙️  Generated {len(recommendations)} fallback recommendations")
        return recommendations[:num_recommendations]
    
    def _format_schema_for_prompt(self, data_profile: DataProfile) -> str:
        """Format data profile as readable schema for LLM"""
        schema_lines = [f"Dataset: {data_profile.dataset_name}"]
        schema_lines.append(f"Total: {data_profile.total_rows} rows, {data_profile.total_columns} columns\n")
        
        for table_name, info in data_profile.tables.items():
            schema_lines.append(f"Table: {table_name}")
            schema_lines.append(f"  Numeric columns: {', '.join(info['numeric'])}")
            schema_lines.append(f"  Categorical columns: {', '.join(info['text'])}")
            schema_lines.append(f"  Date columns: {', '.join(info['date'])}")
            schema_lines.append(f"  Row count: {info['row_count']}\n")
        
        return '\n'.join(schema_lines)
    
    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM JSON response"""
        try:
            if not response_text or response_text.strip() == '':
                logger.error("❌ LLM returned empty response")
                return []
            
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Try to find JSON array anywhere in the text
            if not json_text.startswith('[') and not json_text.startswith('{'):
                # Look for JSON array pattern
                import re
                json_match = re.search(r'\[[\s\S]*\]', response_text)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    # Look for JSON object pattern
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        json_text = json_match.group(0)
            
            logger.debug(f"📋 Extracted JSON text (first 200 chars): {json_text[:200]}")
            
            recommendations = json.loads(json_text)
            
            # Handle both array and object with "recommendations" key
            if isinstance(recommendations, dict) and 'recommendations' in recommendations:
                recommendations = recommendations['recommendations']
            
            if not isinstance(recommendations, list):
                logger.error(f"❌ Expected list, got {type(recommendations)}")
                return []
            
            logger.info(f"✅ Parsed {len(recommendations)} recommendations from LLM")
            return recommendations
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM JSON: {e}")
            logger.error(f"📄 Full response text:\n{response_text}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing LLM response: {e}")
            logger.error(f"📄 Full response text:\n{response_text}")
            return []
    
    def _validate_recommendation(self, rec: Dict[str, Any], data_profile: DataProfile) -> bool:
        """Validate that recommendation uses valid columns"""
        metric = rec.get('metric', '')
        dimension = rec.get('dimension', '')
        
        # Check if columns exist in schema
        for table_info in data_profile.tables.values():
            all_columns = table_info['numeric'] + table_info['text'] + table_info['date']
            if metric in all_columns and dimension in all_columns:
                return True
        
        logger.warning(f"⚠️  Invalid recommendation: {metric} / {dimension} not in schema")
        return False
    
    def _recommendation_to_entities(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Convert recommendation to SmartQueryParser-compatible entities dict"""
        return {
            'metric': rec.get('metric', ''),
            'dimension': rec.get('dimension', ''),
            'chart_type': rec.get('chart_type', 'bar'),
            'aggregation': rec.get('aggregation', 'sum'),
            'filters': rec.get('filters', []),
            'time_period': None,
            'limit': rec.get('limit'),
            'calculation_type': rec.get('calculation_type'),
            'group_by': rec.get('group_by'),
            'time_granularity': rec.get('time_granularity'),
            # Metadata (not used by ChartGenerator, but useful for UI)
            '_title': rec.get('title', ''),
            '_reasoning': rec.get('reasoning', ''),
            '_priority': rec.get('priority', 99)
        }


# ============================================================================
# Component 4: SmartDashboardGenerator (Orchestrator)
# ============================================================================

class SmartDashboardGenerator:
    """
    Main orchestrator for AI-powered dashboard generation
    
    Workflow:
    1. Profile current dataset
    2. Analyze user context
    3. Get LLM recommendations
    4. Generate charts using existing ChartGenerator
    5. Return dashboard with explanations
    """
    
    def __init__(self, data_connector: DataConnector, use_llm: bool = True):
        self.data_connector = data_connector
        self.profiler = DataProfiler(data_connector)
        self.context_analyzer = ContextAnalyzer()
        self.recommender = SmartChartRecommender(use_llm=use_llm)
        self.chart_generator = ChartGenerator(data_connector)
    
    def generate_smart_dashboard(
        self,
        user_department: Optional[str] = None,
        user_role: str = "Viewer",
        override_context: Optional[Dict[str, Any]] = None,
        num_charts: int = 5,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete smart dashboard
        
        Args:
            user_department: User's department (from auth)
            user_role: User's role (Admin, Analyst, etc.)
            override_context: Optional context override
            num_charts: Number of charts to generate
            custom_prompt: Optional custom instructions (e.g., "focus on CEO metrics", "show product X performance")
        
        Returns:
            {
                'success': bool,
                'recommendations': List[Dict],  # Chart recommendations with metadata
                'charts': List[plotly.Figure],  # Generated Plotly charts
                'profile': DataProfile,
                'context': UserContext,
                'error': Optional[str]
            }
        """
        try:
            logger.info("🚀 Starting smart dashboard generation...")
            
            # Step 1: Profile dataset
            logger.info("📊 Step 1: Profiling dataset...")
            data_profile = self.profiler.profile_current_dataset(include_sample=False)
            
            if not data_profile.tables:
                return {
                    'success': False,
                    'error': 'No dataset loaded. Please upload data first.',
                    'recommendations': [],
                    'charts': [],
                    'profile': data_profile,
                    'context': None
                }
            
            # Step 2: Analyze user context
            logger.info("👤 Step 2: Analyzing user context...")
            user_context = self.context_analyzer.analyze_context(
                user_department, user_role, override_context
            )
            
            # Step 3: Get recommendations
            logger.info(f"🤖 Step 3: Generating {num_charts} chart recommendations...")
            recommendations = self.recommender.recommend_charts(
                data_profile, user_context, num_charts, custom_prompt
            )
            
            if not recommendations:
                return {
                    'success': False,
                    'error': 'Could not generate recommendations. Please try manual query.',
                    'recommendations': [],
                    'charts': [],
                    'profile': data_profile,
                    'context': user_context
                }
            
            # Step 4: Generate charts
            logger.info("📈 Step 4: Generating charts...")
            charts = []
            successful_recommendations = []
            
            for i, rec in enumerate(recommendations, 1):
                try:
                    # Create synthetic query for logging
                    query = f"{rec['aggregation']} of {rec['metric']} by {rec['dimension']}"
                    
                    # Generate chart using existing ChartGenerator
                    fig = self.chart_generator.generate_chart(rec, query)
                    
                    if fig:
                        charts.append(fig)
                        successful_recommendations.append(rec)
                        logger.info(f"  ✅ Chart {i}/{len(recommendations)}: {rec.get('_title', query)}")
                    else:
                        logger.warning(f"  ⚠️  Chart {i} generation returned None")
                        
                except Exception as e:
                    logger.error(f"  ❌ Chart {i} failed: {e}")
                    continue
            
            # Step 5: Return results
            logger.info(f"✅ Smart dashboard complete: {len(charts)}/{num_charts} charts generated")
            
            return {
                'success': True,
                'recommendations': successful_recommendations,
                'charts': charts,
                'profile': data_profile,
                'context': user_context,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"❌ Smart dashboard generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': [],
                'charts': [],
                'profile': None,
                'context': None
            }
