# Gen-Dash: AI Coding Agent Instructions

## Quick Reference

Gen-Dash is an **AI-powered dashboard generation platform** with a 4-component agentic AI system that generates context-aware dashboards and converts natural language queries into interactive data visualizations.

**Version**: 1.2.0 (Production-ready refactor)
**Last Update**: February 9, 2026
**Architecture**: Modular FastAPI (9 routers, 57 endpoints, 3 core modules, 4 services)

## 🔥 Recent Major Refactoring - CRITICAL FOR AI AGENTS

**Commit 6b34d27** (Feb 9, 2026): "Main segregated and AgenticAI process"

The codebase underwent a **massive architectural refactoring** to achieve separation of concerns:

**Before**: Monolithic `main.py` (~4,200+ lines) with mixed business logic, routing, and config
**After**: Clean 95-line orchestrator + modular structure with clear boundaries

### Where Everything Moved (Import Changes Required!)

| What | Old Location | New Location | Purpose |
|------|--------------|--------------|---------|
| **API Endpoints** | `main.py` routes | `api/routers/*.py` (9 files) | auth, dashboards, charts, datasets, ratings, exports, explainability, rag, pages |
| **Config & CORS** | `main.py` setup | `core/config.py` | Logging, CORS middleware |
| **Database Pool** | `main.py` setup | `core/database.py` | SQLAlchemy session management |
| **Lifecycle Hooks** | `main.py` events | `core/lifespan.py` | Startup/shutdown logic |
| **Auth Logic** | `main.py` functions | `services/auth_service.py` | Password hashing/verification |
| **Dashboard Singleton** | `main.py` global | `services/dashboard_service.py` | Global dashboard instance |
| **Personalization** | `main.py` stores | `services/feedback_service.py` | Session + time-decay feedback |
| **Shared Dependencies** | Inline in routes | `api/dependencies.py` | FastAPI `@Depends()` patterns |

### Critical Import Changes

```python
# ❌ OLD (pre-Feb 9 refactor) - THESE WILL FAIL
from main import get_dashboard_system
from main import require_auth
from main import hash_password

# ✅ NEW (post-refactor) - USE THESE
from services.dashboard_service import get_dashboard_system
from api.dependencies import require_auth
from services.auth_service import hash_password
```

### How to Navigate Post-Refactor Code

1. **Looking for an endpoint?** → Check `api/routers/{feature}.py`
2. **Looking for business logic?** → Check `services/{feature}_service.py`
3. **Looking for shared code?** → Check `api/dependencies.py` or `core/`
4. **Looking for AI features?** → Still in `dashboard/`, `nlu/`, `charts/` (unchanged)

## 📚 Documentation Structure

This instructions file has been modularized for easier navigation. Load only what you need:

### Core Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture, data flow, global state patterns, RBAC
- **[COMPONENTS.md](COMPONENTS.md)** - Core components (SmartQueryParser, DataConnector, MetricCalculator, Charts)
- **[AI_FEATURES.md](AI_FEATURES.md)** - Smart Dashboard Generator, Dashboard Explainer, Session Feedback, Time-Decay Ratings
- **[API_REFERENCE.md](API_REFERENCE.md)** - API endpoints, request/response patterns
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Environment setup, running the app, database, testing, logging
- **[CONVENTIONS.md](CONVENTIONS.md)** - Coding conventions, common pitfalls, best practices

### Quick Start Guide

```bash
# 1. Activate virtual environment (REQUIRED)
source venv/bin/activate

# 2. Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access: http://localhost:8000
```

## 🎯 Most Common Tasks

### When Working on Query Processing

→ See [COMPONENTS.md](COMPONENTS.md) - SmartQueryParser, Entity Structure

### When Working on AI Features

→ See [AI_FEATURES.md](AI_FEATURES.md) - Smart Generator, Explainer, Personalization

### When Adding API Endpoints

→ See [API_REFERENCE.md](API_REFERENCE.md) + [CONVENTIONS.md](CONVENTIONS.md) - Auth patterns, error handling

### When Debugging

→ See [DEVELOPMENT.md](DEVELOPMENT.md) - Logging, testing, database setup

### When Understanding Architecture

→ See [ARCHITECTURE.md](ARCHITECTURE.md) - Request pipeline, global state, multi-dataset

## 🔑 Critical Patterns (Must Know)

### 1. Global Dashboard System Singleton ⚠️ CRITICAL

```python
# ✅ ALWAYS use this pattern - import from services
from services.dashboard_service import get_dashboard_system
dashboard_system = get_dashboard_system()

# ❌ NEVER instantiate directly - causes chart accumulation bugs
from dashboard.interactive_dashboard import InteractiveDashboard
dashboard_system = InteractiveDashboard()  # WRONG - creates duplicate instance
```

**Why this matters**: Charts accumulate in `dashboard_system.dashboard.charts[]` across requests. Multiple instances = lost charts.

**Chart lifecycle pattern**:

```python
# 1. Get singleton
dashboard_system = get_dashboard_system()

# 2. Clear previous session charts
dashboard_system.dashboard.clear_charts()

# 3. Add charts
dashboard_system.dashboard.add_chart_from_query(query, entities)

# 4. Generate final dashboard
dashboard_system.dashboard.generate_and_save_dashboard(user_id, title)
```

### 2. Entity Structure - Lowercase Keys Required ⚠️

```python
# ✅ CORRECT - flat dict with lowercase string keys
entities = {
    'metric': 'sales',               # String value, not list
    'dimension': 'region',           # Lowercase key
    'chart_type': 'bar',            # bar, line, pie, area, scatter, heatmap, radar
    'aggregation': 'sum',           # sum, avg, count, min, max
    'filters': [],                  # List of filter conditions
    'group_by': 'category',         # Optional secondary dimension
    'time_granularity': 'monthly',  # daily, monthly, quarterly, yearly
    'calculation_type': 'yoy_growth',  # Optional: yoy_growth, mom_change, etc.
}

# ❌ WRONG - common mistakes
entities = {
    'METRIC': 'sales',              # Uppercase keys not recognized
    'metric': ['sales', 'profit'],  # Lists not supported for entity values
    'Dimension': 'region',          # Mixed case fails
}
```

**Used in**: `ChartGenerator.generate_chart(entities, query)`, `SmartQueryParser.parse_query()`, `ChartRecommendation` model

### 3. Agentic AI System - 4-Component Architecture

**Location**: `dashboard/smart_generator.py` (1254 lines)

```python
from dashboard.smart_generator import SmartDashboardGenerator

# Orchestrates 4 components automatically:
generator = SmartDashboardGenerator(data_connector, use_llm=True)
result = generator.generate_smart_dashboard(
    user_department="Finance",
    user_role="Analyst",
    num_charts=5,
    custom_prompt="focus on CEO-level KPIs"  # Optional override
)
# Returns: {success, recommendations, charts, profile, context, failed_recommendations}
```

**4 Components** (orchestrated automatically):

1. **DataProfiler** → Analyzes schema, detects industry, identifies semantic columns
2. **ContextAnalyzer** → Maps department → preferred metrics, role → chart types
3. **SmartChartRecommender** → LLM-powered (Claude 3 Haiku), validates columns with multi-layer fallback
4. **SmartDashboardGenerator** → Orchestrates workflow, generates charts, handles failures

### 4. Dual Personalization System

**Tier 1: Session Feedback** (Ephemeral, RAM-based)

- `services/feedback_service.py` → `session_feedback_store[session_id]`
- Tracks likes/dislikes during active session
- Cleared on logout/timeout
- Used for immediate session personalization

**Tier 2: Time-Decay Ratings** (Persistent, JSON file-based)

- File: `data/user_ratings/user_{id}_ratings.json` (v2 schema)
- Exponential decay: `weight = e^(-0.02 × days_old)` (35-day half-life)
- Survives across sessions
- Used for long-term user preference modeling

### 5. Multi-Dataset Architecture

```python
# DataConnector maintains single active dataset
data_connector.get_available_datasets()     # List all in data/ subdirs
data_connector.switch_dataset('healthcare') # Switch active dataset
data_connector.get_current_dataset()        # Returns 'healthcare'

# Auto-detection (NO hardcoded column names):
# - Numeric: int64, float64 dtypes
# - Categorical: object dtype
# - Date: datetime64 or recognized patterns
```

**File structure**: `data/{dataset_name}/{tables}.xlsx`

## 📦 Tech Stack Summary

| Component       | Technology                                         |
| --------------- | -------------------------------------------------- |
| Backend         | FastAPI (async), uvicorn                           |
| Database        | MySQL + SQLAlchemy ORM (pool_size=5, recycle=3600) |
| Charts          | Plotly (interactive)                               |
| NLP Query       | SmartQueryParser (fuzzy + LLM hybrid)              |
| AI/LLM          | Claude 3 Haiku via OpenRouter                      |
| RAG             | LangChain + FAISS (vector similarity)              |
| Auth            | Bcrypt + Session cookies (30-min timeout)          |
| Personalization | Session store (RAM) + Time-decay (JSON files)      |

## 🚨 Common Mistakes to Avoid

See [CONVENTIONS.md](CONVENTIONS.md) for full list. **Top 9 critical errors**:

1. ❌ **Using old imports from main.py** → ✅ Use new modular imports (see refactoring section above)
2. ❌ Creating multiple `InteractiveDashboard()` instances → ✅ Use `get_dashboard_system()` from `services/`
3. ❌ Forgetting `clear_charts()` before new session → ✅ Always clear to prevent chart accumulation
4. ❌ Using uppercase/mixed-case entity keys → ✅ All keys lowercase: `'metric'` not `'METRIC'`
5. ❌ Entity values as lists → ✅ Always strings: `'metric': 'sales'` not `'metric': ['sales']`
6. ❌ Not activating venv → ✅ Run `source venv/bin/activate` before starting
7. ❌ Missing `OPENROUTER_API_KEY` in `.env` → ✅ Required for all AI features (SmartGenerator, Explainer, QueryParser LLM fallback)
8. ❌ Directly importing models without session management → ✅ Use `@Depends(get_db)` for database operations
9. ❌ Hardcoding column names → ✅ Use DataConnector's auto-detection (adapts to any dataset structure)

## 🔗 Environment Variables

```bash
# Database (MySQL recommended for production)
DATABASE_URL=mysql+pymysql://user:pass@localhost/analytics_dashboard

# AI Features (REQUIRED for SmartGenerator, Explainer, NLP fallback)
OPENROUTER_API_KEY=sk-or-v1-...  # Claude 3 Haiku via OpenRouter

# Auth (JWT)
SECRET_KEY=...                    # For session tokens
```

## 🏗️ Architecture Quick Reference

### Request Pipeline

```
HTTP Request (session_id cookie)
    ↓
@Depends(require_auth) → Session validation + RBAC check
    ↓
@Depends(get_dashboard_system) → Get singleton instance
    ↓
@Depends(get_db) → SQLAlchemy session (connection pool)
    ↓
Router handler → Business logic in services/
    ↓
JSONResponse or HTMLResponse
```

### Chart Generation Flow

```
User Query String
    ↓
SmartQueryParser (fuzzy 70-80% success, LLM fallback 20-30%)
    ↓
entities dict (lowercase keys, flat structure)
    ↓
ChartGenerator.generate_chart(entities, query)
    ↓
MetricCalculator (if calculation_type specified)
    ↓
Plotly figure generation
    ↓
DashboardGenerator.add_chart_from_query()
    ↓
Accumulates in dashboard_system.dashboard.charts[]
```

### Key File Locations (main.py:1-95)

```python
# Entry point - minimal orchestration only
/main.py (95 lines)

# Core infrastructure
/core/config.py      # Logging, CORS
/core/database.py    # SQLAlchemy pool
/core/lifespan.py    # Startup/shutdown

# Business logic
/services/dashboard_service.py  # Singleton management
/services/auth_service.py       # Password hashing
/services/feedback_service.py   # Personalization

# API layer (57 endpoints across 9 routers)
/api/routers/auth.py            # Session management
/api/routers/dashboards.py      # Dashboard CRUD + RBAC
/api/routers/charts.py          # Chart generation
/api/routers/datasets.py        # Data switching
/api/routers/ratings.py         # User feedback
/api/routers/explainability.py  # AI insights
/api/routers/rag.py             # Document Q&A

# AI/ML features
/dashboard/smart_generator.py   # 4-component agentic system
/nlu/smart_query_parser.py      # Hybrid NLP (fuzzy + LLM)
/dashboard/dashboard_explainer.py # AI insights generator

# Data layer
/charts/data_connector.py       # Multi-dataset loader
/charts/chart_generator.py      # Plotly chart creation
/charts/metric_calculator.py    # YoY, MoM, MA calculations
```

## 📝 When to Load Which File

### By Task Type

- **Understanding system design & architecture?** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Working on NLP/query parsing or entity extraction?** → [COMPONENTS.md](COMPONENTS.md)
- **Implementing AI features (SmartGenerator, Explainer, RAG)?** → [AI_FEATURES.md](AI_FEATURES.md)
- **Adding/modifying API endpoints?** → [API_REFERENCE.md](API_REFERENCE.md)
- **Setting up dev environment or debugging?** → [DEVELOPMENT.md](DEVELOPMENT.md)
- **Following code standards or checking conventions?** → [CONVENTIONS.md](CONVENTIONS.md)

### By Specific Feature

| Working on...                        | Read these files                                                            |
| ------------------------------------ | --------------------------------------------------------------------------- |
| Dashboard generation pipeline        | `services/dashboard_service.py`, `dashboard/interactive_dashboard.py:1-300` |
| Agentic AI system                    | `dashboard/smart_generator.py:1-100`, [AI_FEATURES.md](AI_FEATURES.md)      |
| Natural language query parsing       | `nlu/smart_query_parser.py:1-200`, [COMPONENTS.md](COMPONENTS.md)           |
| Chart generation & Plotly            | `charts/chart_generator.py:1-150`, `charts/metric_calculator.py:1-100`      |
| Multi-dataset switching              | `charts/data_connector.py:1-200`                                            |
| Authentication & RBAC                | `api/routers/auth.py`, `auth/auth.py`, [CONVENTIONS.md](CONVENTIONS.md)     |
| Personalization (session/time-decay) | `services/feedback_service.py`, `api/routers/ratings.py`                    |
| AI insights & explanations           | `dashboard/dashboard_explainer.py:1-150`                                    |
| RAG document Q&A                     | `rag/rag.py`, `rag/loaders.py`                                              |

---

## 💡 Quick Tips for AI Agents

1. **Start with the singleton**: Any dashboard work begins with `get_dashboard_system()` from services
2. **Validate entities early**: Check lowercase keys and string values before passing to generators
3. **Check column existence**: Use DataConnector's `get_columns()` to verify columns exist before querying
4. **Enable LLM fallback**: Set `use_llm=True` in SmartQueryParser for better query understanding
5. **Test incrementally**: The system supports adding charts one at a time - test each before proceeding
6. **Leverage RAG for context**: When explaining dashboards, use RAG with company documents for richer insights
7. **Respect RBAC boundaries**: Dashboard visibility depends on role - test with different user roles

## When updating this file

Keep the files segregated, don't populate this file unless required.
Keep all the relevant information in relevant MD files.

**Last Updated**: February 9, 2026 | **Version**: 1.2.0 (Production refactor)
