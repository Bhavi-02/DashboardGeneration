# Gen-Dash: AI Coding Agent Instructions

## Quick Reference

Gen-Dash is an **AI-powered dashboard generation platform** with a 4-component agentic AI system that generates context-aware dashboards and converts natural language queries into interactive data visualizations.

**Version**: 1.2.0 (Production-ready refactor)
**Last Update**: February 10, 2026
**Architecture**: Modular FastAPI (9 routers, 57 endpoints, 3 core modules, 4 services)

---

## When updating this file (🚨 CRITICAL)

Keep this and all .MD files small compact and segregated, don't populate file unless required.
Keep all the relevant information in relevant MD files.
Reference the new file if you made one.

---

## 🚨 CRITICAL: Major Refactoring (Feb 9, 2026)

The codebase underwent a **massive architectural refactoring** moving from a monolithic 4,200+ line `main.py` to a clean modular structure.

**⚠️ Import changes required!** See [REFACTORING.md](REFACTORING.md) for migration guide.

**Quick migration tips**:

```python
# ❌ OLD - THESE WILL FAIL
from main import get_dashboard_system, require_auth, hash_password

# ✅ NEW - USE THESE
from services.dashboard_service import get_dashboard_system
from api.dependencies import require_auth
from services.auth_service import hash_password
```

---

## 📚 Documentation Structure

**Load only what you need based on your task:**

| Documentation                            | Purpose                                          | When to Read                       |
| ---------------------------------------- | ------------------------------------------------ | ---------------------------------- |
| **[REFACTORING.md](REFACTORING.md)**     | Feb 2026 refactoring details                     | Working with pre-refactor code     |
| **[ARCHITECTURE.md](ARCHITECTURE.md)**   | System design, data flow, RBAC, tech stack       | Understanding architecture         |
| **[CONVENTIONS.md](CONVENTIONS.md)**     | Critical patterns, coding standards, pitfalls    | Writing/reviewing code             |
| **[COMPONENTS.md](COMPONENTS.md)**       | SmartQueryParser, DataConnector, Charts          | Query processing, entity structure |
| **[AI_FEATURES.md](AI_FEATURES.md)**     | Smart Generator, Explainer, RAG, Personalization | AI/ML features                     |
| **[API_REFERENCE.md](API_REFERENCE.md)** | Endpoints, auth patterns, RBAC                   | Adding/modifying APIs              |
| **[DEVELOPMENT.md](DEVELOPMENT.md)**     | Setup, running, database, testing, env vars      | Dev environment, debugging         |

---

## 🎯 Quick Start

```bash
# 1. Activate virtual environment (REQUIRED)
source venv/bin/activate

# 2. Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access: http://localhost:8000
```

---

## 🔑 Top 5 Critical Patterns

### 1. Global Dashboard Singleton (⚠️ MOST CRITICAL)

```python
# ✅ ALWAYS use this
from services.dashboard_service import get_dashboard_system
dashboard_system = get_dashboard_system()

# ❌ NEVER do this - causes chart accumulation bugs
dashboard_system = InteractiveDashboard()  # WRONG
```

**Why**: Charts accumulate globally. Multiple instances = lost charts.
**Details**: [CONVENTIONS.md - Critical Patterns](CONVENTIONS.md)

### 2. Entity Structure - Lowercase Keys Only

```python
# ✅ CORRECT
entities = {'metric': 'sales', 'dimension': 'region', 'chart_type': 'bar'}

# ❌ WRONG
entities = {'METRIC': 'sales', 'metric': ['sales']}  # Uppercase or lists fail
```

**Details**: [CONVENTIONS.md - Entity Structure](CONVENTIONS.md)

### 3. Chart Lifecycle Pattern

```python
dashboard_system = get_dashboard_system()
dashboard_system.dashboard.clear_charts()  # ALWAYS clear first
dashboard_system.dashboard.add_chart_from_query(query, entities)
dashboard_system.dashboard.generate_and_save_dashboard(user_id, title)
```

**Details**: [CONVENTIONS.md - Chart Lifecycle](CONVENTIONS.md)

### 4. Module Imports Post-Refactor

- **API Endpoints** → `api/routers/{feature}.py`
- **Business Logic** → `services/{feature}_service.py`
- **Shared Dependencies** → `api/dependencies.py`
- **AI Features** → `dashboard/`, `nlu/`, `charts/` (unchanged)

**Details**: [REFACTORING.md](REFACTORING.md)

### 5. Multi-Dataset Pattern

```python
data_connector.get_available_datasets()     # List all datasets
data_connector.switch_dataset('healthcare') # Switch active
data_connector.get_current_dataset()        # Verify active
```

**Details**: [ARCHITECTURE.md - Multi-Dataset](ARCHITECTURE.md)

---

## 🚨 Top 5 Common Mistakes

1. ❌ Using old imports from `main.py` → ✅ Use new modular imports ([REFACTORING.md](REFACTORING.md))
2. ❌ Creating multiple dashboard instances → ✅ Use `get_dashboard_system()` singleton
3. ❌ Forgetting `clear_charts()` → ✅ Always clear before new session
4. ❌ Uppercase entity keys → ✅ All lowercase: `'metric'` not `'METRIC'`
5. ❌ Not activating venv → ✅ Run `source venv/bin/activate` first

**Full list**: [CONVENTIONS.md - Common Pitfalls](CONVENTIONS.md)

---

## 📝 Task-Based Navigation

### Working on...

| Task                                              | Read These                                                             |
| ------------------------------------------------- | ---------------------------------------------------------------------- |
| **Query parsing / NLP**                           | [COMPONENTS.md](COMPONENTS.md) - SmartQueryParser                      |
| **Chart generation**                              | [COMPONENTS.md](COMPONENTS.md) - ChartGenerator, MetricCalculator      |
| **AI features (Smart Generator, Explainer, RAG)** | [AI_FEATURES.md](AI_FEATURES.md)                                       |
| **API endpoints / Auth**                          | [API_REFERENCE.md](API_REFERENCE.md), [CONVENTIONS.md](CONVENTIONS.md) |
| **Dashboard pipeline**                            | [ARCHITECTURE.md](ARCHITECTURE.md) - Request Pipeline                  |
| **Environment setup / Debugging**                 | [DEVELOPMENT.md](DEVELOPMENT.md)                                       |
| **Code review / Standards**                       | [CONVENTIONS.md](CONVENTIONS.md)                                       |
| **Migration from old code**                       | [REFACTORING.md](REFACTORING.md)                                       |

---

## 💡 Quick Tips for AI Agents

1. **Start with the singleton**: Use `get_dashboard_system()` from `services/` for all dashboard work
2. **Validate entities early**: Lowercase keys + string values before passing to generators
3. **Check columns exist**: Use `DataConnector.get_columns()` before querying
4. **Enable LLM fallback**: Set `use_llm=True` in SmartQueryParser for better accuracy
5. **Test incrementally**: Add charts one at a time, test each before proceeding
6. **Respect RBAC**: Dashboard visibility depends on user role
7. **Reference the right docs**: Don't guess - load the specific .md file for your task

---

## 🏗️ Project Structure Snapshot

```
/main.py (95 lines)          # Minimal orchestrator
/core/                       # Logging, CORS, database, lifecycle
/services/                   # Business logic (dashboard, auth, feedback)
/api/routers/                # 9 routers, 57 endpoints
/api/dependencies.py         # Shared FastAPI dependencies
/dashboard/                  # Smart generator, explainer (AI features)
/nlu/                        # SmartQueryParser (NLP)
/charts/                     # Chart generation, data connector
/database/                   # SQLAlchemy models
/auth/                       # Authentication & RBAC
/rag/                        # RAG system (LangChain + FAISS)
```

**Detailed file locations**: [ARCHITECTURE.md - Key File Locations](ARCHITECTURE.md)

---

## 🔗 External References

- **Tech Stack**: [ARCHITECTURE.md - Tech Stack](ARCHITECTURE.md)
- **Environment Variables**: [DEVELOPMENT.md - Environment Variables](DEVELOPMENT.md)
- **Request Pipeline**: [ARCHITECTURE.md - Request Pipeline](ARCHITECTURE.md)
- **Chart Generation Flow**: [ARCHITECTURE.md - Chart Generation Flow](ARCHITECTURE.md)

---

**Last Updated**: February 10, 2026 | **Version**: 1.2.0 (Production refactor)
