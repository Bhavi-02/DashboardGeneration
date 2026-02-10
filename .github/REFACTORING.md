# Gen-Dash: Major Refactoring History

## 🔥 February 2026 Refactoring - CRITICAL FOR AI AGENTS

**Commit 6b34d27** (Feb 9, 2026): "Main segregated and AgenticAI process"

The codebase underwent a **massive architectural refactoring** to achieve separation of concerns:

**Before**: Monolithic `main.py` (~4,200+ lines) with mixed business logic, routing, and config
**After**: Clean 95-line orchestrator + modular structure with clear boundaries

## Where Everything Moved (Import Changes Required!)

| What                    | Old Location        | New Location                    | Purpose                                                                          |
| ----------------------- | ------------------- | ------------------------------- | -------------------------------------------------------------------------------- |
| **API Endpoints**       | `main.py` routes    | `api/routers/*.py` (9 files)    | auth, dashboards, charts, datasets, ratings, exports, explainability, rag, pages |
| **Config & CORS**       | `main.py` setup     | `core/config.py`                | Logging, CORS middleware                                                         |
| **Database Pool**       | `main.py` setup     | `core/database.py`              | SQLAlchemy session management                                                    |
| **Lifecycle Hooks**     | `main.py` events    | `core/lifespan.py`              | Startup/shutdown logic                                                           |
| **Auth Logic**          | `main.py` functions | `services/auth_service.py`      | Password hashing/verification                                                    |
| **Dashboard Singleton** | `main.py` global    | `services/dashboard_service.py` | Global dashboard instance                                                        |
| **Personalization**     | `main.py` stores    | `services/feedback_service.py`  | Session + time-decay feedback                                                    |
| **Shared Dependencies** | Inline in routes    | `api/dependencies.py`           | FastAPI `@Depends()` patterns                                                    |

## Critical Import Changes

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

## How to Navigate Post-Refactor Code

1. **Looking for an endpoint?** → Check `api/routers/{feature}.py`
2. **Looking for business logic?** → Check `services/{feature}_service.py`
3. **Looking for shared code?** → Check `api/dependencies.py` or `core/`
4. **Looking for AI features?** → Still in `dashboard/`, `nlu/`, `charts/` (unchanged)

## Migration Checklist

If you're updating pre-refactor code:

- [ ] Replace `from main import` with appropriate module imports
- [ ] Use `get_dashboard_system()` from `services/dashboard_service`
- [ ] Use `require_auth` from `api/dependencies`
- [ ] Use auth functions from `services/auth_service`
- [ ] Check if endpoint moved to `api/routers/`
- [ ] Verify database session management uses `@Depends(get_db)`
- [ ] Confirm CORS/config moved to `core/config.py`

**Last Updated**: February 9, 2026
