"""
Smart Query Parser - Hybrid approach with fuzzy matching + LLM fallback
Replaces custom NER model for better flexibility across any dataset
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from fuzzywuzzy import fuzz, process
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


# Pydantic models for structured LLM output
class QueryEntities(BaseModel):
    """Structured query entities extracted from natural language"""
    metric: Optional[str] = Field(description="The numeric metric to measure (e.g., sales, revenue, count)")
    dimension: Optional[str] = Field(description="The categorical dimension to group by (e.g., region, category, month)")
    chart_type: Optional[str] = Field(description="Suggested chart type (bar, line, pie, area, scatter, heatmap, radar)")
    aggregation: Optional[str] = Field(description="Aggregation function (sum, avg, count, min, max)")
    filters: List[str] = Field(default_factory=list, description="Filter conditions from the query")
    time_period: Optional[str] = Field(default=None, description="Time period if mentioned (e.g., 2023, last quarter)")
    limit: Optional[int] = Field(default=None, description="Result limit (e.g., top 10)")
    
    # NEW: Multi-series support for line/area charts
    group_by: Optional[str] = Field(
        default=None,
        description="Secondary categorical dimension for grouping/coloring (e.g., 'product category' in 'sales over time by product category'). Creates multiple lines/series."
    )
    time_granularity: Optional[str] = Field(
        default=None,
        description="Time granularity for time-series: 'daily', 'monthly', 'quarterly', 'yearly'. Use 'quarterly' for seasonal patterns, 'monthly' for trends."
    )
    
    # NEW: Calculated metrics support
    calculation_type: Optional[str] = Field(
        default=None, 
        description="Type of calculation needed: 'yoy_growth' (year-over-year), 'mom_change' (month-over-month), 'percent_change', 'per_unit' (per order/customer), 'cumulative', 'moving_average', or None for simple aggregation"
    )
    calculation_window: Optional[int] = Field(
        default=None,
        description="Window size for moving average calculations (e.g., 3 for 3-month moving average)"
    )
    comparison_type: Optional[str] = Field(
        default=None,
        description="Comparison type: 'vs_previous', 'vs_baseline', 'year_over_year', or None"
    )


@dataclass
class ColumnMatch:
    """Result of column matching with confidence score"""
    column_name: str
    confidence: int  # 0-100
    match_type: str  # 'exact', 'fuzzy', 'llm'


class SmartQueryParser:
    """
    Hybrid query parser using fuzzy matching first, LLM as fallback
    
    Architecture:
    1. Extract available columns from dataset
    2. Try fuzzy matching (fast, 70-80% success rate)
    3. If fuzzy fails or low confidence, use LLM (slower, 95%+ accuracy)
    4. Cache results to reduce API calls
    """
    
    def __init__(self, api_key: Optional[str] = None, use_llm: bool = True):
        """
        Initialize smart query parser
        
        Args:
            api_key: OpenRouter API key (reads from env if not provided)
            use_llm: Enable LLM fallback (set False for fuzzy-only mode)
        """
        self.use_llm = use_llm
        self.cache = {}  # Query -> Entities cache
        
        # Calculation trigger words (minimal hardcoding)
        self.calculation_triggers = [
            'growth', 'change', 'increase', 'decrease', 'decline',
            'per', 'each', 'every', 'average',
            'cumulative', 'total', 'running', 'accumulated',
            'moving', 'rolling', 'trend',
            'compare', 'vs', 'versus', 'against', 'comparison'
        ]
        
        # Initialize LLM if enabled
        if self.use_llm:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                logger.warning("No OPENROUTER_API_KEY found. LLM fallback disabled.")
                self.use_llm = False
            else:
                self.llm = ChatOpenAI(
                    model="anthropic/claude-3-haiku",
                    openai_api_key=self.api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.1,
                    max_tokens=500
                )
                self.parser = PydanticOutputParser(pydantic_object=QueryEntities)
                logger.info("✅ LLM initialized: Claude 3 Haiku via OpenRouter")
        
        # Chart type keywords for fuzzy matching
        self.chart_keywords = {
            'bar': ['bar', 'bars', 'column', 'columns', 'by'],
            'line': ['line', 'trend', 'over time', 'time series', 'timeline'],
            'pie': ['pie', 'distribution', 'proportion', 'percentage', 'share'],
            'area': ['area', 'stacked area', 'cumulative'],
            'scatter': ['scatter', 'correlation', 'relationship', 'vs'],
            'heatmap': ['heatmap', 'heat map', 'matrix', 'correlation matrix'],
            'radar': ['radar', 'spider', 'comparison']
        }
        
        # Aggregation keywords
        self.agg_keywords = {
            'sum': ['total', 'sum', 'overall'],
            'avg': ['average', 'avg', 'mean'],
            'count': ['count', 'number of', 'how many'],
            'min': ['minimum', 'min', 'lowest'],
            'max': ['maximum', 'max', 'highest', 'top']
        }
    
    def _validate_column_type(self, column: str, role: str, available_columns: List[str]) -> bool:
        """Validate if column type is appropriate for its role (metric/dimension)"""
        # This is a basic heuristic - can be enhanced with actual dataframe inspection
        column_lower = column.lower()
        
        if role == 'metric':
            # Metrics should typically be numeric
            numeric_keywords = ['amount', 'sales', 'revenue', 'cost', 'price', 'total', 'sum', 
                               'count', 'quantity', 'value', 'number', 'avg', 'average']
            return any(kw in column_lower for kw in numeric_keywords)
        
        elif role == 'dimension':
            # Dimensions are typically categorical
            categorical_keywords = ['name', 'type', 'category', 'region', 'territory', 
                                   'department', 'product', 'customer', 'key', 'id', 
                                   'date', 'month', 'year', 'day', 'quarter']
            return any(kw in column_lower for kw in categorical_keywords)
        
        return True  # Default to valid
    
    def _extract_limit_filter(self, query: str) -> Optional[int]:
        """Extract limit/top N filter from query"""
        patterns = [
            r'top\s+(\d+)',
            r'first\s+(\d+)',
            r'limit\s+(\d+)',
            r'show\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_time_period_filter(self, query: str) -> Optional[str]:
        """Extract time period from query"""
        query_lower = query.lower()
        
        # Year patterns
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            return f"year={year_match.group(1)}"
        
        # Quarter patterns
        quarter_match = re.search(r'\bq([1-4])\b', query_lower)
        if quarter_match:
            return f"quarter={quarter_match.group(1)}"
        
        # Month patterns
        months = ['january', 'february', 'march', 'april', 'may', 'june', 
                 'july', 'august', 'september', 'october', 'november', 'december']
        for i, month in enumerate(months, 1):
            if month in query_lower:
                return f"month={i}"
        
        # Relative time
        if 'last year' in query_lower:
            return "time=last_year"
        elif 'this year' in query_lower:
            return "time=this_year"
        elif 'last quarter' in query_lower:
            return "time=last_quarter"
        elif 'this quarter' in query_lower:
            return "time=this_quarter"
        elif 'last month' in query_lower:
            return "time=last_month"
        elif 'this month' in query_lower:
            return "time=this_month"
        
        return None
    
    def _extract_range_filter(self, query: str) -> List[str]:
        """Extract range filters from query (e.g., 'between X and Y', 'greater than X')"""
        filters = []
        query_lower = query.lower()
        
        # Between X and Y
        between_match = re.search(r'between\s+([\d.]+)\s+and\s+([\d.]+)', query_lower)
        if between_match:
            filters.append(f"range={between_match.group(1)}-{between_match.group(2)}")
        
        # Greater than / More than
        gt_match = re.search(r'(?:greater|more)\s+than\s+([\d.]+)', query_lower)
        if gt_match:
            filters.append(f"min={gt_match.group(1)}")
        
        # Less than / Fewer than
        lt_match = re.search(r'(?:less|fewer)\s+than\s+([\d.]+)', query_lower)
        if lt_match:
            filters.append(f"max={lt_match.group(1)}")
        
        return filters
    
    def parse_query(self, query: str, available_columns: List[str], column_samples: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Parse natural language query into structured entities
        
        Args:
            query: Natural language query (e.g., "sales by region")
            available_columns: List of column names from the dataset
            column_samples: Optional dict with column names as keys and list of sample values (head 5)
        
        Returns:
            Dictionary with extracted entities (metric, dimension, chart_type, etc.)
        """
        # Check cache first
        cache_key = f"{query}|{','.join(sorted(available_columns))}"
        if cache_key in self.cache:
            logger.info(f"✅ Cache hit for query: {query}")
            return self.cache[cache_key]
        
        query_lower = query.lower().strip()
        
        # Check if query needs calculation (use LLM for these)
        needs_calculation = any(
            trigger in query_lower 
            for trigger in self.calculation_triggers
        )
        
        if needs_calculation:
            logger.info(f"🧮 Calculation detected in query, routing to LLM: {query}")
        
        # Step 1: Try fuzzy matching (fast path) - ONLY for non-calculation queries
        fuzzy_result = None
        if not needs_calculation:
            fuzzy_result = self._fuzzy_parse(query_lower, available_columns)
            
            # If fuzzy matching has high confidence, use it
            if fuzzy_result and self._is_high_confidence(fuzzy_result):
                logger.info(f"✅ Fuzzy match succeeded for: {query}")
                self.cache[cache_key] = fuzzy_result
                return fuzzy_result
        
        # Step 2: LLM for calculation queries OR low-confidence fuzzy matches
        if self.use_llm:
            logger.info(f"🤖 Using LLM fallback for: {query}")
            llm_result = self._llm_parse(query, available_columns, column_samples)
            
            if llm_result:
                logger.info(f"✅ LLM parse succeeded for: {query}")
                self.cache[cache_key] = llm_result
                return llm_result
        
        # Step 3: Fallback to fuzzy result even if low confidence
        if fuzzy_result:
            logger.warning(f"⚠️ Using low-confidence fuzzy match for: {query}")
            self.cache[cache_key] = fuzzy_result
            return fuzzy_result
        
        # Step 4: Return empty result (graceful degradation)
        logger.error(f"❌ Failed to parse query: {query}")
        return self._empty_result()
    
    def _fuzzy_parse(self, query: str, columns: List[str]) -> Optional[Dict[str, Any]]:
        """
        Fuzzy matching approach for common query patterns
        
        Strategy:
        - Match column names using fuzzy string matching (fuzzywuzzy)
        - Detect chart types from keywords
        - Detect aggregation functions from keywords
        - Handle common patterns like "X by Y", "total X", "average Y"
        """
        result = {
            'metric': None,
            'dimension': None,
            'chart_type': 'bar',  # Default
            'aggregation': 'sum',  # Default
            'filters': [],
            'time_period': None,
            'limit': None,
            'match_confidence': 0
        }
        
        # Detect chart type from keywords
        result['chart_type'] = self._detect_chart_type(query)
        
        # Detect aggregation from keywords
        result['aggregation'] = self._detect_aggregation(query)
        
        # Extract filters
        limit = self._extract_limit_filter(query)
        if limit:
            result['limit'] = limit
            result['filters'].append(f'top {limit}')
        
        time_period = self._extract_time_period_filter(query)
        if time_period:
            result['time_period'] = time_period
            result['filters'].append(time_period)
        
        range_filters = self._extract_range_filter(query)
        if range_filters:
            result['filters'].extend(range_filters)
        
        # Match columns using fuzzy matching
        # Pattern 1: "metric by dimension" (e.g., "sales by region")
        if ' by ' in query:
            parts = query.split(' by ')
            metric_query = parts[0].strip()
            dimension_query = parts[1].strip()
            
            # Match metric
            metric_match = self._fuzzy_match_column(metric_query, columns)
            if metric_match:
                # Validate column type for metric role
                if self._validate_column_type(metric_match.column_name, 'metric', columns):
                    result['metric'] = metric_match.column_name
                    result['match_confidence'] += metric_match.confidence / 2
                else:
                    logger.warning(f"Column '{metric_match.column_name}' may not be suitable as metric")
                    result['metric'] = metric_match.column_name  # Use it anyway but log warning
                    result['match_confidence'] += (metric_match.confidence * 0.7) / 2  # Reduce confidence
            
            # Match dimension
            dimension_match = self._fuzzy_match_column(dimension_query, columns)
            if dimension_match:
                # Validate column type for dimension role
                if self._validate_column_type(dimension_match.column_name, 'dimension', columns):
                    result['dimension'] = dimension_match.column_name
                    result['match_confidence'] += dimension_match.confidence / 2
                else:
                    logger.warning(f"Column '{dimension_match.column_name}' may not be suitable as dimension")
                    result['dimension'] = dimension_match.column_name  # Use it anyway but log warning
                    result['match_confidence'] += (dimension_match.confidence * 0.7) / 2  # Reduce confidence
        
        # Pattern 2: Single column (e.g., "sales over time", "revenue trend")
        elif any(kw in query for kw in ['over time', 'trend', 'timeline']):
            # Extract main column and assume time-based dimension
            words = query.replace('over time', '').replace('trend', '').replace('timeline', '').strip()
            metric_match = self._fuzzy_match_column(words, columns)
            if metric_match:
                result['metric'] = metric_match.column_name
                result['match_confidence'] = metric_match.confidence
                result['chart_type'] = 'line'  # Time series use line charts
                
                # Try to find a date column
                date_columns = [col for col in columns if any(date_kw in col.lower() 
                               for date_kw in ['date', 'time', 'month', 'year', 'day'])]
                if date_columns:
                    result['dimension'] = date_columns[0]
        
        # Pattern 3: "total/average/count of X" or just "X"
        else:
            # Remove aggregation keywords
            cleaned_query = query
            for agg_type, keywords in self.agg_keywords.items():
                for kw in keywords:
                    cleaned_query = cleaned_query.replace(kw, '').strip()
            
            # Remove chart type keywords
            for chart_type, keywords in self.chart_keywords.items():
                for kw in keywords:
                    cleaned_query = cleaned_query.replace(kw, '').strip()
            
            # Match remaining words to columns
            metric_match = self._fuzzy_match_column(cleaned_query, columns)
            if metric_match:
                result['metric'] = metric_match.column_name
                result['match_confidence'] = metric_match.confidence
        
        # Return None if no matches found
        if not result['metric'] and not result['dimension']:
            return None
        
        return result
    
    def _llm_parse(self, query: str, columns: List[str], column_samples: Optional[Dict[str, List[Any]]] = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        LLM-based parsing using Claude 3 Haiku with retry logic
        
        Uses structured output (Pydantic) to ensure consistent format
        Implements exponential backoff for transient errors
        """
        last_error = None
        
        # Format column information with sample data if available
        if column_samples:
            columns_info = "Available columns with sample data:\n"
            for col in columns:
                if col in column_samples and column_samples[col]:
                    # Format sample values
                    samples = column_samples[col][:5]  # Max 5 samples
                    sample_str = ", ".join(str(v) for v in samples if v is not None)
                    columns_info += f"  - {col}: {sample_str}\n"
                else:
                    columns_info += f"  - {col}\n"
        else:
            columns_info = f"Available columns: {', '.join(columns)}"
        
        for attempt in range(max_retries):
            try:
                # Create prompt with available columns and strict JSON output requirement
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a data analysis assistant that extracts structured information from natural language queries.

{columns_info}

Your task:
1. Identify the METRIC (numeric column to measure) - look at sample data to identify numeric columns
2. Identify the DIMENSION (primary axis for grouping) - **FOR LINE CHARTS: prioritize time columns (Date, Month, Quarter, FY)**
3. Identify GROUP_BY (secondary dimension for multi-series charts) - categorical column that creates multiple lines/colors
4. Suggest appropriate CHART_TYPE (bar, line, pie, area, scatter, heatmap, radar)
5. Determine TIME_GRANULARITY for time-series (daily, monthly, quarterly, yearly)
6. Determine AGGREGATION function (sum, avg, count, min, max)
7. Extract any FILTERS mentioned (time periods, limits, ranges)
8. Detect CALCULATION_TYPE if query needs calculations

IMPORTANT: Look at the sample data values to understand what each column contains. Match the user's intent to the most appropriate column based on the actual data.

**TABLEAU-STYLE DIMENSION PRIORITY:**
- For **LINE** or **AREA** charts:
  - DIMENSION should be time-based (Date, Month, Quarter, FY) if available
  - GROUP_BY should be categorical (Product Category, State, etc.)
  - Example: "sales trends by product category" → dimension='Month', group_by='Product Category', chart_type='line'

- For **BAR** or **PIE** charts:
  - DIMENSION should be categorical
  - GROUP_BY is usually None (unless stacked bars)
  - Example: "sales by region" → dimension='State', group_by=None, chart_type='bar'

**TIME PHRASE DETECTION - CRITICAL PATTERN:**
If query contains time phrases like "over time", "across years", "over the years", "through time", "year by year", "historically":
  → DIMENSION MUST be a time column (Date, Month, Quarter, FY, Year)
  → If query also asks "which X" or "compare X", then X becomes GROUP_BY (multi-series)
  → Use LINE or GROUPED BAR chart for better comparison visualization

**TIME_GRANULARITY DETECTION:**
- "seasonal" / "seasonality" / "seasonal patterns" → time_granularity='quarterly'
- "monthly trends" / "month by month" / "each month" → time_granularity='monthly'
- "yearly" / "annual" / "by year" / "across years" / "over the years" / "year by year" → time_granularity='yearly'
- "daily" / "day by day" → time_granularity='daily'

**MULTI-SERIES EXAMPLES:**
- "Are there seasonal spikes in sales for certain product categories?"
  → metric='Net Amount', dimension='Quarter', group_by='Product Category', chart_type='line', time_granularity='quarterly'

- "Monthly sales trends by region"
  → metric='Net Amount', dimension='Month', group_by='State', chart_type='line', time_granularity='monthly'

- "Which branch generated higher sales across the years?"
  → metric='Taxable Amt', dimension='FY', group_by='Location', chart_type='line', time_granularity='yearly'
  (Note: "which branch" = comparing branches, "across years" = time dimension)

- "Compare product categories over time"
  → metric='Net Amount', dimension='Date', group_by='Product Category', chart_type='line'

- "Sales by product category" (NO time-series)
  → metric='Net Amount', dimension='Product Category', group_by=None, chart_type='bar'

**CALCULATION DETECTION:**
- "year-on-year" / "yoy" → calculation_type='yoy_growth'
- "month-on-month" / "mom" → calculation_type='mom_change'
- "per order" / "per customer" → calculation_type='per_unit'
- "cumulative" / "running total" → calculation_type='cumulative'
- "moving average" / "rolling average" → calculation_type='moving_average', calculation_window=3
- Regular aggregation ONLY → calculation_type=None

CRITICAL: You MUST respond with ONLY the JSON object, no explanations, no additional text before or after.

{format_instructions}"""),
                    ("user", "{query}")
                ])
                
                # Format prompt
                formatted_prompt = prompt.format_messages(
                    query=query,
                    columns_info=columns_info,
                    format_instructions=self.parser.get_format_instructions()
                )
                
                # Call LLM
                response = self.llm.invoke(formatted_prompt)
                
                # Parse response
                entities = self.parser.parse(response.content)
                
                # Convert to dictionary and validate columns
                result = {
                    'metric': self._validate_column(entities.metric, columns),
                    'dimension': self._validate_column(entities.dimension, columns),
                    'chart_type': entities.chart_type or 'bar',
                    'aggregation': entities.aggregation or 'sum',
                    'filters': entities.filters or [],
                    'time_period': entities.time_period,
                    'limit': entities.limit,
                    'group_by': self._validate_column(entities.group_by, columns) if entities.group_by else None,  # NEW
                    'time_granularity': entities.time_granularity,  # NEW
                    'calculation_type': entities.calculation_type,
                    'calculation_window': entities.calculation_window,
                    'comparison_type': entities.comparison_type,
                    'match_confidence': 95  # LLM typically has high confidence
                }
                
                # Validate column types
                if result['metric'] and not self._validate_column_type(result['metric'], 'metric', columns):
                    logger.warning(f"LLM suggested metric '{result['metric']}' may not be numeric")
                    result['match_confidence'] = 85  # Lower confidence
                
                if result['dimension'] and not self._validate_column_type(result['dimension'], 'dimension', columns):
                    logger.warning(f"LLM suggested dimension '{result['dimension']}' may not be categorical")
                    result['match_confidence'] = 85  # Lower confidence
                
                return result
            
            except ValueError as e:
                # JSON parsing or Pydantic validation error
                logger.error(f"LLM response parsing error (attempt {attempt + 1}/{max_retries}): {e}")
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                continue
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for specific error types
                if 'api key' in error_msg or 'authentication' in error_msg or 'unauthorized' in error_msg:
                    logger.error(f"API authentication error: {e}")
                    logger.error("Please check your OPENROUTER_API_KEY in .env file")
                    return None  # Don't retry auth errors
                
                elif 'rate limit' in error_msg or '429' in error_msg:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 * (2 ** attempt)  # Longer wait for rate limits
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                
                elif 'timeout' in error_msg or 'connection' in error_msg:
                    logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1 * (2 ** attempt))  # Exponential backoff
                        continue
                
                else:
                    logger.error(f"LLM parsing error (attempt {attempt + 1}/{max_retries}): {e}")
                
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
        
        # All retries exhausted
        logger.error(f"LLM parsing failed after {max_retries} attempts. Last error: {last_error}")
        return None
    
    def _fuzzy_match_column(self, query: str, columns: List[str]) -> Optional[ColumnMatch]:
        """
        Fuzzy match query text to available columns
        
        Uses fuzzywuzzy library with token_set_ratio for flexible matching
        Returns best match if confidence > 60
        """
        if not query or not columns:
            return None
        
        # Use extractOne for best match
        best_match = process.extractOne(
            query,
            columns,
            scorer=fuzz.token_set_ratio
        )
        
        if best_match and best_match[1] > 60:  # Confidence threshold
            return ColumnMatch(
                column_name=best_match[0],
                confidence=best_match[1],
                match_type='fuzzy'
            )
        
        return None
    
    def _validate_column(self, column: Optional[str], available_columns: List[str]) -> Optional[str]:
        """Validate and fuzzy-match column name from LLM output"""
        if not column:
            return None
        
        # If exact match, return it
        if column in available_columns:
            return column
        
        # Otherwise, try fuzzy match
        match = self._fuzzy_match_column(column, available_columns)
        return match.column_name if match else None
    
    def _detect_chart_type(self, query: str) -> str:
        """Detect chart type from query keywords"""
        for chart_type, keywords in self.chart_keywords.items():
            if any(kw in query for kw in keywords):
                return chart_type
        return 'bar'  # Default
    
    def _detect_aggregation(self, query: str) -> str:
        """Detect aggregation function from query keywords"""
        for agg_type, keywords in self.agg_keywords.items():
            if any(kw in query for kw in keywords):
                return agg_type
        return 'sum'  # Default
    
    def _is_high_confidence(self, result: Dict[str, Any]) -> bool:
        """Check if fuzzy match result has high confidence (>80)"""
        return result.get('match_confidence', 0) > 80
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for failed parsing"""
        return {
            'metric': None,
            'dimension': None,
            'chart_type': 'bar',
            'aggregation': 'sum',
            'filters': [],
            'time_period': None,
            'limit': None,
            'match_confidence': 0
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.cache.clear()
        logger.info("Query cache cleared")


# Backward compatibility function for old NER-based code
def migrate_ner_entities_to_smart_parser(ner_entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert old NER entity format to new smart parser format
    
    Used for auto-migration of existing saved dashboards
    
    Old format (NER):
    {
        'METRIC': ['sales'],
        'DIMENSION': ['region'],
        'CHART_TYPE': ['bar'],
        'AGGREGATION': ['sum'],
        'FILTER': []
    }
    
    New format (Smart Parser):
    {
        'metric': 'sales',
        'dimension': 'region',
        'chart_type': 'bar',
        'aggregation': 'sum',
        'filters': []
    }
    """
    return {
        'metric': ner_entities.get('METRIC', [None])[0],
        'dimension': ner_entities.get('DIMENSION', [None])[0],
        'chart_type': ner_entities.get('CHART_TYPE', ['bar'])[0],
        'aggregation': ner_entities.get('AGGREGATION', ['sum'])[0],
        'filters': ner_entities.get('FILTER', []),
        'match_confidence': 100  # Assume old NER results were confident
    }
