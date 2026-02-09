"""Chart ratings and feedback API router for Gen-Dash"""
from fastapi import APIRouter, Depends, Body, Cookie
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
from services.feedback_service import _chart_signature, _calculate_decay_weight
from api.dependencies import require_auth
import logging
import traceback
import json
import uuid
import shutil

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/rate-chart")
async def rate_chart(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """
    Unified chart rating endpoint (replaces save-chart-favorites and record-chart-feedback).

    Supports three-state ratings: 'like', 'neutral', 'dislike'
    - Recent ratings weighted more heavily (time-decayed preferences)
    - Click same button to toggle back to neutral
    - Persists to user profile for cross-session personalization

    Request Body:
    {
        "chart": {
            "signature": "Net Amount|Location|line|||",
            "metric": "Net Amount",
            "dimension": "Location",
            "chart_type": "line",
            "title": "Sales by Branch",
            "reasoning": "..."
        },
        "rating": "like" | "neutral" | "dislike"
    }
    """
    try:
        chart = request_data.get('chart')
        rating = request_data.get('rating')

        # Validation
        if not chart or not isinstance(chart, dict):
            return JSONResponse({
                "success": False,
                "message": "chart object is required"
            }, status_code=400)

        if rating not in ['like', 'neutral', 'dislike']:
            return JSONResponse({
                "success": False,
                "message": "rating must be 'like', 'neutral', or 'dislike'"
            }, status_code=400)

        user_id = session['user_id']
        username = session['username']
        signature = chart.get('signature', _chart_signature(chart))

        # Load or create user ratings file
        ratings_dir = Path("data/user_ratings")
        ratings_dir.mkdir(parents=True, exist_ok=True)
        ratings_file = ratings_dir / f"user_{user_id}_ratings.json"

        if ratings_file.exists():
            with open(ratings_file, 'r') as f:
                user_data = json.load(f)
        else:
            user_data = {
                "user_id": user_id,
                "username": username,
                "department": session.get('department'),
                "role": session.get('role'),
                "version": 2,
                "ratings": [],
                "statistics": {"total_ratings": 0, "likes": 0, "dislikes": 0, "neutral": 0}
            }

        # Find existing rating or create new
        existing_idx = None
        for idx, r in enumerate(user_data['ratings']):
            if r['signature'] == signature:
                existing_idx = idx
                break

        timestamp = datetime.now(timezone.utc).isoformat()

        if existing_idx is not None:
            # Update existing rating
            old_rating = user_data['ratings'][existing_idx]['rating']
            user_data['ratings'][existing_idx]['rating'] = rating
            user_data['ratings'][existing_idx]['timestamp'] = timestamp
            user_data['ratings'][existing_idx]['rating_history'].append({
                "rating": rating,
                "timestamp": timestamp
            })

            # Update statistics
            if old_rating != rating:
                user_data['statistics'][old_rating + 's'] -= 1
                user_data['statistics'][rating + 's'] += 1

            rating_id = user_data['ratings'][existing_idx]['rating_id']
        else:
            # Create new rating
            rating_id = str(uuid.uuid4())

            user_data['ratings'].append({
                "rating_id": rating_id,
                "signature": signature,
                "rating": rating,
                "chart_metadata": {
                    "metric": chart.get('metric'),
                    "dimension": chart.get('dimension'),
                    "chart_type": chart.get('chart_type'),
                    "title": chart.get('title'),
                    "reasoning": chart.get('reasoning')
                },
                "timestamp": timestamp,
                "rating_history": [{"rating": rating, "timestamp": timestamp}]
            })

            user_data['statistics']['total_ratings'] += 1
            user_data['statistics'][rating + 's'] += 1

        # Remove neutral ratings (clean up - neutral is absence of opinion)
        user_data['ratings'] = [r for r in user_data['ratings'] if r['rating'] != 'neutral']

        # Update metadata
        user_data['last_updated'] = timestamp
        user_data['statistics']['last_activity'] = timestamp

        # Save to file
        with open(ratings_file, 'w') as f:
            json.dump(user_data, f, indent=2)

        logger.info(f"{'❤️' if rating == 'like' else '👎' if rating == 'dislike' else '➖'} User {username} rated chart as '{rating}': {chart.get('title')}")

        return JSONResponse({
            "success": True,
            "message": f"Chart rated as '{rating}'",
            "rating": {
                "rating_id": rating_id,
                "signature": signature,
                "rating": rating,
                "timestamp": timestamp
            },
            "user_stats": {
                "total_ratings": user_data['statistics']['total_ratings'],
                "likes": user_data['statistics']['likes'],
                "dislikes": user_data['statistics']['dislikes']
            }
        })

    except Exception as e:
        logger.error(f"❌ Error saving chart rating: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.get("/get-chart-ratings")
async def get_chart_ratings(session: dict = Depends(require_auth)):
    """
    Get user's chart ratings with time-decay weights.

    Returns all user's ratings (likes and dislikes) with decay weights calculated
    based on age of rating. Recent ratings have weight ~1.0, older ratings decay
    exponentially (35-day half-life).

    Response:
    {
        "success": true,
        "user_id": 1,
        "last_updated": "2026-02-08T10:30:00Z",
        "ratings": [
            {
                "signature": "Net Amount|Location|line|||",
                "rating": "like",
                "chart_metadata": {...},
                "timestamp": "2026-02-08T10:15:30Z",
                "decay_weight": 0.98
            }
        ],
        "statistics": {"total_ratings": 25, "likes": 12, "dislikes": 3}
    }
    """
    try:
        user_id = session['user_id']
        ratings_file = Path("data/user_ratings") / f"user_{user_id}_ratings.json"

        if not ratings_file.exists():
            return JSONResponse({
                "success": True,
                "user_id": user_id,
                "ratings": [],
                "statistics": {"total_ratings": 0, "likes": 0, "dislikes": 0}
            })

        with open(ratings_file, 'r') as f:
            user_data = json.load(f)

        # Add decay weights to each rating
        for rating in user_data['ratings']:
            rating['decay_weight'] = _calculate_decay_weight(rating['timestamp'])

        return JSONResponse({
            "success": True,
            "user_id": user_id,
            "last_updated": user_data.get('last_updated'),
            "ratings": user_data['ratings'],
            "statistics": user_data['statistics']
        })

    except Exception as e:
        logger.error(f"❌ Error loading chart ratings: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/clear-chart-preferences")
async def clear_chart_preferences_api(
    session: dict = Depends(require_auth)
):
    """
    Clear all user chart preferences/ratings to reset personalization.
    This allows users to start fresh with chart recommendations.

    Returns:
    {
        "success": true,
        "message": "Preferences cleared successfully",
        "user_id": 1
    }
    """
    try:
        user_id = session['user_id']
        ratings_file = Path("data/user_ratings") / f"user_{user_id}_ratings.json"

        if ratings_file.exists():
            # Backup the file before deletion (optional safety measure)
            backup_dir = Path("data/user_ratings/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / f"user_{user_id}_ratings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            shutil.copy(ratings_file, backup_file)
            logger.info(f"📋 Backed up ratings to {backup_file}")

            # Delete the ratings file
            ratings_file.unlink()
            logger.info(f"🗑️  Cleared preferences for user {user_id}")
            message = "All preferences cleared successfully. Backup saved."
        else:
            logger.info(f"ℹ️  No preferences file found for user {user_id}")
            message = "No preferences to clear."

        return JSONResponse({
            "success": True,
            "message": message,
            "user_id": user_id
        })

    except Exception as e:
        logger.error(f"❌ Error clearing chart preferences: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# ============================================================================
# Legacy Endpoints (Deprecated - kept for backward compatibility)
# ============================================================================
# TODO: Remove after frontend migration complete (v2.1.0)

@router.post("/save-chart-favorites")
async def save_chart_favorites_legacy(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """
    DEPRECATED: Use /api/rate-chart instead

    Legacy endpoint maintained for backward compatibility.
    Will be removed in v2.1.0
    """
    logger.warning("⚠️ Legacy endpoint /api/save-chart-favorites called - migrate to /api/rate-chart")

    try:
        favorites = request_data.get('favorites', [])

        # Convert favorites to new rating format
        responses = []
        for fav in favorites:
            chart_data = {
                "signature": _chart_signature(fav),
                "metric": fav.get('metric'),
                "dimension": fav.get('dimension'),
                "chart_type": fav.get('chart_type'),
                "title": fav.get('title'),
                "reasoning": fav.get('reasoning')
            }

            # Call new unified endpoint
            response = await rate_chart(
                request_data={"chart": chart_data, "rating": "like"},
                session=session
            )
            responses.append(response)

        return JSONResponse({
            "success": True,
            "message": f"Saved {len(favorites)} favorites (via legacy endpoint)",
            "favorites_count": len(favorites),
            "warning": "This endpoint is deprecated. Please migrate to /api/rate-chart"
        })

    except Exception as e:
        logger.error(f"❌ Error in legacy favorites endpoint: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/record-chart-feedback")
async def record_chart_feedback_legacy(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth),
    session_id: Optional[str] = Cookie(None)
):
    """
    DEPRECATED: Use /api/rate-chart instead

    Legacy endpoint maintained for backward compatibility.
    Will be removed in v2.1.0
    """
    logger.warning("⚠️ Legacy endpoint /api/record-chart-feedback called - migrate to /api/rate-chart")

    try:
        chart = request_data.get('chart')
        liked = request_data.get('liked')

        if not chart:
            return JSONResponse({
                "success": False,
                "error": "chart is required"
            }, status_code=400)

        # Convert to new rating format
        rating = "like" if liked else "dislike"

        # Call new unified endpoint
        return await rate_chart(
            request_data={"chart": chart, "rating": rating},
            session=session
        )

    except Exception as e:
        logger.error(f"❌ Error in legacy feedback endpoint: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
