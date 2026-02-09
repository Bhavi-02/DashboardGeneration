# Gen-Dash: AI Coding Agent Instructions

## Quick Reference

Gen-Dash is an **AI-powered dashboard generation platform** that converts natural language queries into interactive data visualizations.

**Version**: 1.2.0 (December 19, 2025)  
**Last Update**: February 9, 2026

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

### Global Dashboard System

```python
# Always use get_dashboard_system() - never create new instances
dashboard_system = get_dashboard_system()
dashboard_system.dashboard.clear_charts()  # Clear before new sessions
```

### Entity Structure (Query Processing)

```python
entities = {
    'metric': 'sales',           # Lowercase keys
    'dimension': 'region',       # String values (not lists)
    'chart_type': 'bar',        # bar, line, pie, area, scatter, heatmap, radar
    'aggregation': 'sum',       # sum, avg, count, min, max
}
```

### Dual Personalization System

- **Session Feedback** (RAM): Ephemeral, current session only
- **Time-Decay Ratings** (JSON): Persistent, exponential decay (λ=0.02, 35-day half-life)

### Multi-Dataset Pattern

```python
data_connector.get_available_datasets()     # List all datasets
data_connector.switch_dataset('healthcare') # Switch active dataset
data_connector.get_current_dataset()        # Verify current dataset
```

## 📦 Tech Stack Summary

| Component | Technology                            |
| --------- | ------------------------------------- |
| Backend   | FastAPI (async)                       |
| Database  | MySQL + SQLAlchemy ORM                |
| Charts    | Plotly (interactive)                  |
| NLP       | SmartQueryParser (fuzzy + LLM hybrid) |
| AI/LLM    | Claude 3 Haiku via OpenRouter         |
| RAG       | LangChain + FAISS                     |
| Auth      | Bcrypt + Session-based                |

## 🚨 Common Mistakes to Avoid

See [CONVENTIONS.md](CONVENTIONS.md) for full list. Top 5:

1. ❌ Creating multiple `InteractiveDashboard()` instances → ✅ Use `get_dashboard_system()`
2. ❌ Forgetting to clear charts → ✅ Call `dashboard_system.dashboard.clear_charts()`
3. ❌ Using uppercase entity keys → ✅ All keys lowercase: `'metric'` not `'METRIC'`
4. ❌ Not activating venv → ✅ Run `source venv/bin/activate` first
5. ❌ Missing `OPENROUTER_API_KEY` → ✅ Set in `.env` for AI features

## 🔗 Environment Variables

```bash
DATABASE_URL=mysql+pymysql://user:pass@localhost/analytics_dashboard
OPENROUTER_API_KEY=sk-or-...  # Required for AI features (Claude 3 Haiku)
SECRET_KEY=...                 # For JWT tokens
```

## 📝 When to Load Which File

- **Understanding system design?** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Working on NLP/query parsing?** → [COMPONENTS.md](COMPONENTS.md)
- **Implementing AI features?** → [AI_FEATURES.md](AI_FEATURES.md)
- **Adding/modifying APIs?** → [API_REFERENCE.md](API_REFERENCE.md)
- **Setting up dev environment?** → [DEVELOPMENT.md](DEVELOPMENT.md)
- **Following code standards?** → [CONVENTIONS.md](CONVENTIONS.md)

---

**Last Updated**: February 9, 2026 | **Version**: 1.2.0
