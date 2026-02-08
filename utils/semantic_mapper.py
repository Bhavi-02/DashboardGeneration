"""
Semantic Mapper for Smart Dashboard Generation

Maps business/semantic terms to actual dataset column names using:
1. User-defined mappings (from config/semantic_mappings.json)
2. Auto-detected mappings (pattern-based heuristics)
"""

import json
import os
import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dashboard.smart_generator import DataProfile

logger = logging.getLogger(__name__)


class SemanticMapper:
    """
    Maps semantic/business terms to actual column names

    Features:
    - User-defined mappings (JSON config)
    - Auto-detected mappings (heuristic-based)
    - Multi-dataset support
    """

    def __init__(self, data_profile: Optional["DataProfile"] = None, config_path: Optional[str] = None):
        """
        Initialize semantic mapper

        Args:
            data_profile: DataProfile object with schema information
            config_path: Path to JSON configuration file with mappings
        """
        self.data_profile = data_profile
        self.mappings: Dict[str, str] = {}  # {semantic_term: actual_column}

        # Load user-defined mappings from config (if exists)
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    # Filter out metadata fields (starting with _)
                    self.mappings = {k: v for k, v in config_data.items() if not k.startswith('_')}
                logger.info(f"✅ Loaded {len(self.mappings)} semantic mappings from {config_path}")
            except Exception as e:
                logger.error(f"❌ Error loading semantic mappings from {config_path}: {e}")

        # Auto-detect mappings using heuristics (if data_profile provided)
        if data_profile:
            self._auto_detect_mappings()

    def _auto_detect_mappings(self):
        """Auto-detect semantic mappings based on column content and patterns"""
        if not self.data_profile:
            return

        detected_count = 0

        for table_name, info in self.data_profile.tables.items():
            all_columns = info['numeric'] + info['text'] + info['date']

            for col in all_columns:
                col_lower = col.lower().strip()

                # Only auto-detect if not already in user-defined mappings
                # This allows user mappings to take precedence

                # Financial metrics
                if 'sales' not in self.mappings and any(keyword in col_lower for keyword in ['amount', 'total', 'value', 'sales']):
                    if col not in self.mappings.values():  # Avoid duplicate mappings
                        self.mappings['sales'] = col
                        self.mappings['revenue'] = col
                        detected_count += 1

                # Geographic dimensions
                if 'region' not in self.mappings and any(keyword in col_lower for keyword in ['state', 'region', 'territory', 'location', 'branch']):
                    if col not in self.mappings.values():
                        self.mappings['region'] = col
                        self.mappings['location'] = col
                        self.mappings['area'] = col
                        detected_count += 1

                # Product dimensions
                if 'product' not in self.mappings and any(keyword in col_lower for keyword in ['product', 'category', 'item']):
                    if col not in self.mappings.values():
                        self.mappings['product'] = col
                        self.mappings['item'] = col
                        detected_count += 1

                # Time dimensions
                if 'year' not in self.mappings and any(keyword in col_lower for keyword in ['year', 'fy', 'period', 'fiscal']):
                    if col not in self.mappings.values():
                        self.mappings['year'] = col
                        self.mappings['period'] = col
                        detected_count += 1

                # Quantity metrics
                if 'quantity' not in self.mappings and any(keyword in col_lower for keyword in ['quantity', 'qty', 'units']):
                    if col not in self.mappings.values():
                        self.mappings['quantity'] = col
                        self.mappings['units'] = col
                        detected_count += 1

        if detected_count > 0:
            logger.info(f"📋 Auto-detected {detected_count} additional semantic mappings")

    def map_term(self, term: str) -> Optional[str]:
        """
        Map semantic term to actual column name

        Args:
            term: Business/semantic term to map

        Returns:
            Actual column name if mapping exists, None otherwise
        """
        if not term:
            return None

        term_lower = term.lower().strip()

        # Check exact mapping
        if term_lower in self.mappings:
            mapped = self.mappings[term_lower]
            logger.debug(f"🔗 Mapped '{term}' → '{mapped}'")
            return mapped

        # Check if term is already a valid column (no mapping needed)
        if self.data_profile:
            for info in self.data_profile.tables.values():
                all_columns = info['numeric'] + info['text'] + info['date']
                if term in all_columns:
                    logger.debug(f"✅ '{term}' is already a valid column")
                    return term  # Already valid, no mapping needed

        logger.debug(f"⚠️ No mapping found for '{term}'")
        return None

    def add_mapping(self, semantic_term: str, actual_column: str):
        """
        Add or update a custom mapping

        Args:
            semantic_term: Business/semantic term
            actual_column: Actual column name in dataset
        """
        self.mappings[semantic_term.lower().strip()] = actual_column
        logger.info(f"➕ Added mapping: '{semantic_term}' → '{actual_column}'")

    def save_mappings(self, config_path: str):
        """
        Save current mappings to JSON file for reuse

        Args:
            config_path: Path to save configuration file
        """
        try:
            # Add metadata
            output_data = {
                "_comment": "Business term → Actual column name mappings",
                "_last_updated": "auto-generated",
                **self.mappings
            }

            with open(config_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"💾 Saved {len(self.mappings)} mappings to {config_path}")
        except Exception as e:
            logger.error(f"❌ Error saving mappings to {config_path}: {e}")

    def get_all_mappings(self) -> Dict[str, str]:
        """Return all current mappings"""
        return self.mappings.copy()

    def get_mapped_columns(self) -> list:
        """Return list of all actual columns that are mapped to"""
        return list(set(self.mappings.values()))
