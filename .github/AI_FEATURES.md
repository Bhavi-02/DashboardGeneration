# Gen-Dash AI-Powered Features

## Smart Dashboard Generator (Auto-Generation)

**Location**: [dashboard/smart_generator.py](dashboard/smart_generator.py)

**v1.2.0+ Feature**: Zero-query dashboard generation using LLM-powered context analysis

### Architecture

```python
# 4-Component Pipeline
DataProfiler → ContextAnalyzer → SmartChartRecommender → SmartDashboardGenerator
```

### How It Works

1. **DataProfiler**: Analyzes dataset schema (numeric/text/date columns, cardinality)
2. **ContextAnalyzer**: Builds user context (department-specific metrics, role preferences)
3. **SmartChartRecommender**: LLM generates 5 chart recommendations via structured output
4. **SmartDashboardGenerator**: Orchestrates workflow, generates charts, adds to dashboard

### Usage Pattern

```python
smart_gen = SmartDashboardGenerator(data_connector, use_llm=True)
result = smart_gen.generate_smart_dashboard(
    user_department="Finance",
    user_role="Analyst",
    override_context=None,  # Optional custom context
    num_charts=5,
    custom_prompt="focus on CEO metrics"  # Optional: custom LLM instructions
)
# Returns: {success: bool, recommendations: List[dict], charts: List[Figure], profile: DataProfile}
```

### LLM Prompt Strategy

- Provides dataset schema (columns, types, row counts) as context
- Adds department-specific context (e.g., Finance → revenue, cost; Marketing → campaigns, conversion)
- **Custom Prompt Support**: Users can add special instructions like "give charts for CEO" or "focus on product X"
- Uses Pydantic `structured_output` for type-safe JSON responses
- Model: Claude 3 Haiku via OpenRouter (requires `OPENROUTER_API_KEY`)

### API Endpoint

`POST /api/generate-smart-dashboard`:

```json
{
  "num_charts": 5,
  "custom_prompt": "focus on executive-level metrics",
  "override_context": {
    "department": "Sales",
    "role": "Manager"
  }
}
```

**Critical**: Clears existing charts via `dashboard_system.dashboard.clear_charts()` before generation

## Dashboard Explainer (AI Insights)

**Location**: [dashboard/dashboard_explainer.py](dashboard/dashboard_explainer.py)

**v1.2.0 Major Update**: Replaced ALL hardcoded templates with LLM-generated explanations

### Two Operating Modes

1. **Chart-Only Mode** (default when no documents provided):
   - AI analyzes chart data structure, metrics, dimensions
   - Generates comprehensive insights: trends, comparisons, actionable recommendations
   - Methods: `_explain_charts_only()`, `_generate_chart_only_explanation()`, `_extract_chart_insights()`

2. **RAG-Enhanced Mode** (when documents uploaded):
   - Combines chart analysis + company documents (PDFs/DOCX/PPTX)
   - Uses FAISS vector store for semantic search
   - Provides context-aware insights linked to business documents

### Key Methods

```python
explain_dashboard(queries, force_chart_only=False)  # Main entry point
load_company_profile(file_path)                    # Load docs for RAG
explain_single_chart(query, entities)              # Single chart explanation
get_comparative_insights(queries_list)             # Multi-chart comparison
```

### LLM Configuration

- Model: Claude 3 Haiku (via OpenRouter)
- Temperature: 0.3 (focused but slightly creative)
- Max tokens: 800
- Requires: `OPENROUTER_API_KEY` environment variable

### API Endpoints

- `POST /api/explain-dashboard`: `{queries: [], force_chart_only: bool}`
- `POST /api/load-company-profile`: Upload PDF/DOCX/PPTX for RAG context

## Session-Based Feedback System

**Location**: [main.py:L38-L80](main.py#L38-L80)

**Architecture**: In-memory session-level preference tracking for real-time personalization

```python
# Global session feedback store
session_feedback_store = {
    session_id: {
        "likes": [chart_signature, ...],      # Liked chart signatures
        "dislikes": [chart_signature, ...],    # Disliked chart signatures
        "counts": {
            "chart_type": Counter(),           # Preferred chart types
            "metric": Counter(),               # Preferred metrics
            "dimension": Counter()             # Preferred dimensions
        }
    }
}
```

### Key Functions

```python
_chart_signature(chart)                # Creates stable chart ID: "metric|dimension|chart_type|..."
_get_or_create_session_profile(sid)    # Initialize or retrieve session feedback profile
_build_session_prompt(profile)         # Generate LLM prompt snippet from session preferences
```

### Workflow

1. User likes/dislikes charts during session → stored in `session_feedback_store`
2. Smart dashboard generator requests → includes `_build_session_prompt()` output
3. LLM receives context: "Preferred chart types (session): bar, line; Avoid: [disliked signatures]"
4. Session cleared on logout or timeout (30 minutes)

**Critical**: Session feedback is **ephemeral** (RAM only) and complements the **persistent** time-decay rating system

## Time-Decay Rating System

**Location**: [main.py:L86-L180](main.py#L86-L180)

**v2 Architecture**: Unified rating system with exponential time decay for long-term personalization

### Decay Formula

```python
weight = e^(-λ * days_old)
λ = 0.02  # 35-day half-life (balanced approach)

# Weight Examples:
# 1 day old:   98% weight
# 1 week old:  87% weight
# 1 month old: 55% weight
# 3 months:    17% weight
# 6 months:     3% weight
```

### Key Functions

```python
_calculate_decay_weight(timestamp, decay_rate=0.02)  # Time-decay weight calculation
_aggregate_weighted_ratings(ratings)                 # Aggregate ratings with time weights
_build_personalization_prompt(user_id)               # Generate LLM prompt from user's historical ratings
```

### Data Schema

Stored in `data/user_ratings/user_{id}_ratings.json`:

```json
{
  "user_id": 123,
  "username": "analyst1",
  "version": 2,
  "ratings": [
    {
      "id": "uuid",
      "rating": "like",
      "timestamp": "2025-01-15T10:30:00Z",
      "signature": "sales|region|bar|sum||",
      "chart_metadata": {
        "metric": "sales",
        "dimension": "region",
        "chart_type": "bar"
      }
    }
  ]
}
```

### Usage Pattern

1. User rates chart (like/dislike) → saved to JSON file with timestamp
2. System aggregates ratings with time decay weights
3. LLM receives context: "Preferred chart types: bar (weight: 5.2), line (weight: 3.1)"
4. More recent preferences automatically weighted higher

### Critical Difference

- **Session Feedback**: Temporary, resets on logout, for immediate session personalization
- **Time-Decay Ratings**: Persistent, file-based, for long-term user personalization across sessions

## Structured Output Pattern

Use Pydantic models with `structured_output` for type-safe LLM responses:

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class ChartRecommendation(BaseModel):
    metric: str = Field(description="Column to measure")
    dimension: str = Field(description="Column to group by")
    chart_type: str = Field(description="bar, line, pie...")

llm = ChatOpenAI(model="anthropic/claude-3-haiku", temperature=0.3)
structured_llm = llm.with_structured_output(ChartRecommendation)
response = structured_llm.invoke(prompt)  # Returns ChartRecommendation instance
```

**Benefits**: Guarantees valid JSON, eliminates parsing errors, provides autocomplete
