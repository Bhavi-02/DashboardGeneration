# User Chart Favorites

This directory stores user preferences for AI-generated charts, enabling personalized dashboard recommendations.

## Purpose

When users mark charts as "favorites" in the Smart Dashboard, the system records:

- Which chart types users prefer (bar, line, pie, etc.)
- Which metrics they care about (sales, revenue, quantity, etc.)
- Which dimensions they analyze (region, category, time periods, etc.)
- Context: user's department and role

## Data Format

Files are stored as JSONL (JSON Lines) format: `user_{user_id}_favorites.jsonl`

Each line contains:

```json
{
  "user_id": 1,
  "username": "john_doe",
  "department": "Finance",
  "role": "Analyst",
  "timestamp": "2026-01-23T12:00:00Z",
  "favorites": [
    {
      "chart_index": 0,
      "metric": "sales",
      "dimension": "region",
      "chart_type": "bar",
      "title": "Sales by Region",
      "reasoning": "Shows regional performance clearly"
    }
  ],
  "favorite_count": 1
}
```

## Future Enhancements

1. **Personalized Recommendations**: Analyze favorites to suggest similar charts
2. **User Clustering**: Group users with similar preferences
3. **ML Training**: Train models to predict which charts users will like
4. **Department Patterns**: Identify department-wide chart preferences
5. **Smart Prompt Generation**: Auto-generate custom prompts based on favorites

## Analytics Queries

To analyze user preferences:

```python
import json
from pathlib import Path

# Load all favorites
favorites_dir = Path('data/user_favorites')
all_favorites = []

for file in favorites_dir.glob('user_*_favorites.jsonl'):
    with open(file) as f:
        for line in f:
            all_favorites.append(json.loads(line))

# Most popular chart types
chart_types = [f['chart_type'] for fav in all_favorites for f in fav['favorites']]
print(f"Most popular chart types: {Counter(chart_types).most_common(5)}")

# Most popular metrics
metrics = [f['metric'] for fav in all_favorites for f in fav['favorites']]
print(f"Most popular metrics: {Counter(metrics).most_common(5)}")
```

## Privacy & Security

- Favorites are stored per-user and not shared between users
- Data is used solely for improving dashboard recommendations
- No sensitive business data is stored (only chart metadata)
- Users can request deletion of their preferences data
