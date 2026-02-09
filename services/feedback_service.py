"""Session feedback service for chart personalization"""
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter, defaultdict
import json

logger = logging.getLogger(__name__)

# In-memory session feedback store
# Structure: {session_id: {"likes": [sig], "dislikes": [sig], "counts": {"chart_type": Counter, ...}}}
session_feedback_store: Dict[str, Dict[str, Any]] = {}

def _chart_signature(chart: Dict[str, Any]) -> str:
    """Create a stable signature for a chart based on key attributes.

    Args:
        chart: Chart metadata dictionary

    Returns:
        String signature for the chart
    """
    return "|".join([
        str(chart.get('metric', '')).strip(),
        str(chart.get('dimension', '')).strip(),
        str(chart.get('chart_type', '')).strip(),
        str(chart.get('aggregation', '')).strip(),
        str(chart.get('group_by', '')).strip(),
        str(chart.get('calculation_type', '')).strip()
    ])

def _get_or_create_session_profile(session_id: str) -> Dict[str, Any]:
    """Get or initialize the in-memory feedback profile for a session.

    Args:
        session_id: Session identifier

    Returns:
        Session profile dictionary
    """
    if session_id not in session_feedback_store:
        session_feedback_store[session_id] = {
            "likes": [],
            "dislikes": [],
            "counts": {
                "chart_type": Counter(),
                "metric": Counter(),
                "dimension": Counter()
            }
        }
    return session_feedback_store[session_id]

def _build_session_prompt(profile: Dict[str, Any]) -> str:
    """Build a short prompt snippet from session feedback profile.

    Args:
        profile: Session profile dictionary

    Returns:
        Formatted prompt string for LLM
    """
    counts = profile.get("counts", {})
    top_chart_types = [c for c, _ in counts.get("chart_type", Counter()).most_common(3)]
    top_metrics = [m for m, _ in counts.get("metric", Counter()).most_common(3)]
    top_dimensions = [d for d, _ in counts.get("dimension", Counter()).most_common(3)]
    disliked_sigs = profile.get("dislikes", [])[-5:]

    return (
        "\n\nSESSION FEEDBACK CONTEXT:\n"
        f"- Preferred chart types (session): {', '.join(top_chart_types) or 'N/A'}\n"
        f"- Preferred metrics (session): {', '.join(top_metrics) or 'N/A'}\n"
        f"- Preferred dimensions (session): {', '.join(top_dimensions) or 'N/A'}\n"
        f"- Avoid charts similar to these signatures: {', '.join(disliked_sigs) or 'N/A'}\n"
        "Use this to bias recommendations and avoid disliked chart patterns."
    )

def _calculate_decay_weight(rating_timestamp: str, decay_rate: float = 0.02) -> float:
    """Calculate time-decay weight for a rating.

    Formula: weight = e^(-λ * days_old)
    λ = 0.02 (35-day half-life - balanced approach)

    Weight Examples:
    - 1 day old: 98% weight
    - 1 week old: 87% weight
    - 1 month old: 55% weight
    - 3 months old: 17% weight
    - 6 months old: 3% weight

    Args:
        rating_timestamp: ISO format timestamp
        decay_rate: Decay rate lambda (default 0.02)

    Returns:
        Weight in range [0.01, 1.0]
    """
    try:
        rating_time = datetime.fromisoformat(rating_timestamp.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        days_old = (current_time - rating_time).total_seconds() / 86400.0
        weight = math.exp(-decay_rate * days_old)
        return max(0.01, min(1.0, weight))
    except Exception as e:
        logger.warning(f"Error calculating decay weight: {e}")
        return 1.0  # Default to full weight on error

def _aggregate_weighted_ratings(ratings: List[Dict]) -> Dict[str, Any]:
    """Aggregate ratings with time-decay weights.

    Args:
        ratings: List of rating entries with 'rating', 'timestamp', 'chart_metadata'

    Returns:
        dict: {
            'liked_chart_types': [(type, weight), ...],
            'liked_metrics': [(metric, weight), ...],
            'liked_dimensions': [(dimension, weight), ...],
            'disliked_signatures': [signature, ...]
        }
    """
    liked_chart_types = defaultdict(float)
    liked_metrics = defaultdict(float)
    liked_dimensions = defaultdict(float)
    disliked_signatures = []

    for rating_entry in ratings:
        weight = _calculate_decay_weight(rating_entry['timestamp'])
        chart = rating_entry['chart_metadata']
        rating = rating_entry['rating']

        if rating == 'like':
            liked_chart_types[chart.get('chart_type', '')] += weight
            liked_metrics[chart.get('metric', '')] += weight
            liked_dimensions[chart.get('dimension', '')] += weight
        elif rating == 'dislike':
            disliked_signatures.append(rating_entry['signature'])

    return {
        'liked_chart_types': sorted(liked_chart_types.items(), key=lambda x: x[1], reverse=True),
        'liked_metrics': sorted(liked_metrics.items(), key=lambda x: x[1], reverse=True),
        'liked_dimensions': sorted(liked_dimensions.items(), key=lambda x: x[1], reverse=True),
        'disliked_signatures': disliked_signatures[-10:]  # Last 10 dislikes
    }

def _build_personalization_prompt(user_id: int) -> str:
    """Build LLM prompt enrichment from time-decayed user ratings.

    This function loads the user's unified ratings file (v2 schema) and
    aggregates their preferences with time-decay weighting. Recent ratings
    are weighted more heavily than older ratings.

    Args:
        user_id: Database user ID

    Returns:
        str: Formatted prompt snippet to append to custom_prompt, or empty string if no ratings
    """
    try:
        ratings_file = Path("data/user_ratings") / f"user_{user_id}_ratings.json"
        if not ratings_file.exists():
            return ""

        with open(ratings_file, 'r') as f:
            data = json.load(f)

        ratings = data.get('ratings', [])
        if not ratings:
            return ""

        # Aggregate with time decay
        prefs = _aggregate_weighted_ratings(ratings)

        # Build prompt
        prompt_parts = ["\n\n📊 USER PERSONALIZATION CONTEXT:"]

        if prefs['liked_chart_types']:
            top_types = prefs['liked_chart_types'][:3]
            type_list = ', '.join([f"{t} (weight: {w:.1f})" for t, w in top_types])
            prompt_parts.append(f"✅ Preferred chart types: {type_list}")

        if prefs['liked_metrics']:
            top_metrics = prefs['liked_metrics'][:5]
            metric_list = ', '.join([f"{m} (weight: {w:.1f})" for m, w in top_metrics])
            prompt_parts.append(f"✅ Preferred metrics: {metric_list}")

        if prefs['liked_dimensions']:
            top_dims = prefs['liked_dimensions'][:5]
            dim_list = ', '.join([f"{d} (weight: {w:.1f})" for d, w in top_dims])
            prompt_parts.append(f"✅ Preferred dimensions: {dim_list}")

        if prefs['disliked_signatures']:
            sig_list = ', '.join(prefs['disliked_signatures'])
            prompt_parts.append(f"❌ AVOID patterns: {sig_list}")

        prompt_parts.extend([
            "",
            "💡 INSTRUCTIONS:",
            "- Strongly favor chart types and metrics with high weights",
            "- Avoid generating charts matching disliked signatures",
            "- Balance personalization with data-driven insights"
        ])

        return '\n'.join(prompt_parts)

    except Exception as e:
        logger.warning(f"⚠️ Failed to build personalization prompt: {e}")
        return ""
