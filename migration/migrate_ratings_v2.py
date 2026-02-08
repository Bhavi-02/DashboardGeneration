#!/usr/bin/env python3
"""
Migrate old user_favorites and user_feedback data to new unified rating schema.

This script converts the legacy dual-system (favorites + feedback) into a
unified three-state rating system (like/neutral/dislike) with support for
time-decayed preferences.
"""
import json
import os
import uuid
from pathlib import Path
from datetime import datetime
import argparse


def create_signature(chart):
    """Create chart signature from chart metadata."""
    return "|".join([
        str(chart.get('metric', '')),
        str(chart.get('dimension', '')),
        str(chart.get('chart_type', '')),
        str(chart.get('aggregation', '')),
        str(chart.get('group_by', '')),
        str(chart.get('calculation_type', ''))
    ])


def migrate_user(user_id, dry_run=False):
    """Migrate data for a single user."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating user {user_id}...")

    old_favorites_file = Path(f"data/user_favorites/user_{user_id}_favorites.json")
    old_feedback_files = list(Path("data/user_feedback/").glob(f"user_{user_id}_session_*.jsonl"))
    new_ratings_file = Path(f"data/user_ratings/user_{user_id}_ratings.json")

    new_ratings = {
        "user_id": user_id,
        "version": 2,
        "ratings": [],
        "statistics": {"total_ratings": 0, "likes": 0, "dislikes": 0, "neutral": 0}
    }

    signatures_seen = set()

    # Migrate favorites (likes)
    if old_favorites_file.exists():
        print(f"  ✓ Found favorites file")
        with open(old_favorites_file, 'r') as f:
            favorites = json.load(f)

        new_ratings["username"] = favorites.get('username')
        new_ratings["department"] = favorites.get('department')
        new_ratings["role"] = favorites.get('role')

        for fav in favorites.get('favorites', []):
            signature = create_signature(fav)
            if signature not in signatures_seen:
                new_ratings['ratings'].append({
                    "rating_id": str(uuid.uuid4()),
                    "signature": signature,
                    "rating": "like",
                    "chart_metadata": {
                        "metric": fav.get('metric'),
                        "dimension": fav.get('dimension'),
                        "chart_type": fav.get('chart_type'),
                        "title": fav.get('title'),
                        "reasoning": fav.get('reasoning')
                    },
                    "timestamp": favorites.get('timestamp', datetime.now().isoformat()),
                    "rating_history": [{
                        "rating": "like",
                        "timestamp": favorites.get('timestamp', datetime.now().isoformat())
                    }]
                })
                signatures_seen.add(signature)
                new_ratings['statistics']['likes'] += 1
                new_ratings['statistics']['total_ratings'] += 1

        print(f"    → Migrated {new_ratings['statistics']['likes']} favorites")

    # Migrate feedback (dislikes only)
    dislike_count = 0
    for feedback_file in old_feedback_files:
        print(f"  ✓ Found feedback file: {feedback_file.name}")
        with open(feedback_file, 'r') as f:
            for line in f:
                try:
                    feedback = json.loads(line)
                    if not feedback.get('liked', True):  # Dislike
                        signature = feedback.get('signature', create_signature(feedback['chart']))
                        if signature not in signatures_seen:
                            new_ratings['ratings'].append({
                                "rating_id": str(uuid.uuid4()),
                                "signature": signature,
                                "rating": "dislike",
                                "chart_metadata": feedback['chart'],
                                "timestamp": feedback['timestamp'],
                                "rating_history": [{
                                    "rating": "dislike",
                                    "timestamp": feedback['timestamp']
                                }]
                            })
                            signatures_seen.add(signature)
                            dislike_count += 1
                            new_ratings['statistics']['dislikes'] += 1
                            new_ratings['statistics']['total_ratings'] += 1
                except json.JSONDecodeError:
                    continue

    if dislike_count > 0:
        print(f"    → Migrated {dislike_count} dislikes from feedback files")

    # Write new file
    new_ratings['last_updated'] = datetime.now().isoformat()
    new_ratings['statistics']['last_activity'] = datetime.now().isoformat()

    if not dry_run:
        os.makedirs("data/user_ratings", exist_ok=True)
        with open(new_ratings_file, 'w') as f:
            json.dump(new_ratings, f, indent=2)
        print(f"  ✅ Saved to {new_ratings_file}")
    else:
        print(f"  [DRY RUN] Would save to {new_ratings_file}")

    return new_ratings['statistics']


def main():
    parser = argparse.ArgumentParser(description='Migrate chart ratings to v2 schema')
    parser.add_argument('--dry-run', action='store_true', help='Preview migration without writing files')
    parser.add_argument('--user-id', type=int, help='Migrate specific user only')
    args = parser.parse_args()

    print("=" * 60)
    print("Gen-Dash Rating System Migration v2")
    print("=" * 60)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")

    # Find all users
    favorites_dir = Path("data/user_favorites")
    if not favorites_dir.exists():
        print("❌ No favorites directory found")
        return

    user_ids = set()
    for file in favorites_dir.glob("user_*_favorites.json"):
        try:
            user_id = int(file.stem.split('_')[1])
            user_ids.add(user_id)
        except (ValueError, IndexError):
            continue

    if args.user_id:
        user_ids = {args.user_id}

    if not user_ids:
        print("❌ No users found to migrate")
        return

    print(f"Found {len(user_ids)} user(s) to migrate: {sorted(user_ids)}")

    total_stats = {"likes": 0, "dislikes": 0, "total_ratings": 0}

    for user_id in sorted(user_ids):
        stats = migrate_user(user_id, dry_run=args.dry_run)
        total_stats["likes"] += stats["likes"]
        total_stats["dislikes"] += stats["dislikes"]
        total_stats["total_ratings"] += stats["total_ratings"]

    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total users migrated: {len(user_ids)}")
    print(f"Total ratings: {total_stats['total_ratings']}")
    print(f"  - Likes: {total_stats['likes']}")
    print(f"  - Dislikes: {total_stats['dislikes']}")

    if not args.dry_run:
        print("\n✅ Migration complete!")
        print("\n💡 Next steps:")
        print("   1. Test the new /api/get-chart-ratings endpoint")
        print("   2. Test the new /api/rate-chart endpoint")
        print("   3. Deploy updated frontend")
        print("   4. (Optional) Archive old data:")
        print("      mv data/user_favorites data/user_favorites_backup")
        print("      mv data/user_feedback data/user_feedback_backup")
    else:
        print("\n💡 Run without --dry-run to perform actual migration")


if __name__ == '__main__':
    main()
