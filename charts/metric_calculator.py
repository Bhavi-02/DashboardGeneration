"""
Metric Calculator - Handles calculated metrics like YoY growth, per-unit calculations, etc.
Separate from DataConnector for clean separation of concerns
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MetricCalculator:
    """
    Calculates derived metrics from base data
    
    Supports:
    - Year-over-year (YoY) growth
    - Month-over-month (MoM) change
    - Per-unit calculations (per order, per customer, etc.)
    - Cumulative totals
    - Moving averages
    - Percent change
    """
    
    def __init__(self):
        """Initialize metric calculator"""
        pass
    
    def calculate_metric(self, data: pd.DataFrame, calculation_type: str, 
                        metric_col: str, dimension_col: str, 
                        calculation_window: Optional[int] = None) -> pd.DataFrame:
        """
        Main entry point for calculating derived metrics
        
        Args:
            data: Base DataFrame with aggregated data
            calculation_type: Type of calculation ('yoy_growth', 'mom_change', etc.)
            metric_col: Column containing the metric values
            dimension_col: Column containing the dimension (usually time-based)
            calculation_window: Window size for moving average calculations
            
        Returns:
            DataFrame with additional calculated columns
        """
        if data is None or data.empty:
            logger.warning("Empty data provided to MetricCalculator")
            return data
        
        logger.info(f"📊 Calculating {calculation_type} for {metric_col} by {dimension_col}")
        
        # Make a copy to avoid modifying original
        result_df = data.copy()
        
        # Route to appropriate calculation method
        if calculation_type == 'yoy_growth':
            result_df = self.calculate_yoy_growth(result_df, metric_col, dimension_col)
        elif calculation_type == 'mom_change':
            result_df = self.calculate_mom_change(result_df, metric_col, dimension_col)
        elif calculation_type == 'percent_change':
            result_df = self.calculate_percent_change(result_df, metric_col, dimension_col)
        elif calculation_type == 'cumulative':
            result_df = self.calculate_cumulative(result_df, metric_col, dimension_col)
        elif calculation_type == 'moving_average':
            result_df = self.calculate_moving_average(result_df, metric_col, dimension_col, calculation_window or 3)
        elif calculation_type == 'per_unit':
            # Per-unit is handled differently (needs unit column)
            logger.warning("Per-unit calculation requires additional unit column - handled separately")
        else:
            logger.warning(f"Unknown calculation type: {calculation_type}")
        
        return result_df
    
    def calculate_yoy_growth(self, data: pd.DataFrame, metric_col: str, time_col: str) -> pd.DataFrame:
        """
        Calculate year-over-year growth percentage
        
        Formula: YoY Growth % = ((Current Year - Previous Year) / Previous Year) * 100
        
        Args:
            data: DataFrame with time-series data (must be sorted by time)
            metric_col: Column containing metric values
            time_col: Column containing time dimension (year, FY, etc.)
            
        Returns:
            DataFrame with added 'YoY_Growth_Pct' column
        
        Note: Handles both single-series and multi-series (grouped) data
        """
        result_df = data.copy()
        
        # Check if this is multi-series data (has group_by column)
        group_by_cols = [col for col in result_df.columns if col.startswith('group_by_')]
        
        if group_by_cols:
            # Multi-series: Calculate YoY separately for each group
            group_col = group_by_cols[0]
            logger.info(f"📊 Multi-series YoY calculation for each group in {group_col}")
            
            # Sort by group and time
            result_df = result_df.sort_values([group_col, time_col])
            
            # Calculate YoY growth within each group
            result_df['Previous_Value'] = result_df.groupby(group_col)[metric_col].shift(1)
            
            result_df['YoY_Growth_Pct'] = result_df.apply(
                lambda row: ((row[metric_col] - row['Previous_Value']) / row['Previous_Value'] * 100) 
                if row['Previous_Value'] != 0 and pd.notna(row['Previous_Value'])
                else np.nan,
                axis=1
            )
            
            result_df = result_df.drop('Previous_Value', axis=1)
            
            logger.info(f"✅ Multi-series YoY Growth calculated for {result_df[group_col].nunique()} groups")
        else:
            # Single-series: Original logic
            # Sort by time dimension to ensure correct ordering
            result_df = self._sort_by_time_dimension(result_df, time_col)
            
            # Calculate year-over-year change
            result_df['Previous_Value'] = result_df[metric_col].shift(1)
            
            # Calculate growth percentage
            # Handle division by zero: if previous value is 0, set growth to NaN
            result_df['YoY_Growth_Pct'] = result_df.apply(
                lambda row: ((row[metric_col] - row['Previous_Value']) / row['Previous_Value'] * 100) 
                if row['Previous_Value'] != 0 and pd.notna(row['Previous_Value'])
                else np.nan,
                axis=1
            )
            
            # Remove intermediate column
            result_df = result_df.drop('Previous_Value', axis=1)
            
            # Log results
            logger.info(f"✅ YoY Growth calculated:")
            logger.info(f"   Years: {result_df[time_col].tolist()}")
            logger.info(f"   Values: {result_df[metric_col].tolist()}")
            logger.info(f"   Growth %: {[f'{x:.1f}%' if pd.notna(x) else 'N/A' for x in result_df['YoY_Growth_Pct'].tolist()]}")
        
        return result_df
    
    def calculate_mom_change(self, data: pd.DataFrame, metric_col: str, time_col: str) -> pd.DataFrame:
        """
        Calculate month-over-month change percentage
        
        Similar to YoY but for monthly periods
        """
        result_df = data.copy()
        
        # Sort by time dimension
        result_df = self._sort_by_time_dimension(result_df, time_col)
        
        # Calculate month-over-month change
        result_df['Previous_Value'] = result_df[metric_col].shift(1)
        
        result_df['MoM_Change_Pct'] = result_df.apply(
            lambda row: ((row[metric_col] - row['Previous_Value']) / row['Previous_Value'] * 100) 
            if row['Previous_Value'] != 0 and pd.notna(row['Previous_Value'])
            else np.nan,
            axis=1
        )
        
        result_df = result_df.drop('Previous_Value', axis=1)
        
        logger.info(f"✅ MoM Change calculated for {len(result_df)} periods")
        
        return result_df
    
    def calculate_percent_change(self, data: pd.DataFrame, metric_col: str, time_col: str) -> pd.DataFrame:
        """
        Calculate percentage change from first value (baseline)
        
        Formula: % Change = ((Current - Baseline) / Baseline) * 100
        """
        result_df = data.copy()
        
        # Sort by time dimension
        result_df = self._sort_by_time_dimension(result_df, time_col)
        
        # Get baseline (first value)
        baseline_value = result_df[metric_col].iloc[0]
        
        if baseline_value == 0:
            logger.warning("Baseline value is 0, cannot calculate percent change")
            result_df['Percent_Change'] = np.nan
        else:
            result_df['Percent_Change'] = ((result_df[metric_col] - baseline_value) / baseline_value * 100)
        
        logger.info(f"✅ Percent change calculated from baseline: {baseline_value:,.0f}")
        
        return result_df
    
    def calculate_cumulative(self, data: pd.DataFrame, metric_col: str, time_col: str) -> pd.DataFrame:
        """
        Calculate cumulative sum over time
        
        Args:
            data: DataFrame with time-series data
            metric_col: Column to calculate cumulative sum for
            time_col: Time dimension column
        """
        result_df = data.copy()
        
        # Sort by time dimension
        result_df = self._sort_by_time_dimension(result_df, time_col)
        
        # Calculate cumulative sum
        result_df['Cumulative_Total'] = result_df[metric_col].cumsum()
        
        logger.info(f"✅ Cumulative total calculated: {result_df['Cumulative_Total'].iloc[-1]:,.0f} (final)")
        
        return result_df
    
    def calculate_moving_average(self, data: pd.DataFrame, metric_col: str, 
                                 time_col: str, window: int = 3) -> pd.DataFrame:
        """
        Calculate moving/rolling average
        
        Args:
            data: DataFrame with time-series data
            metric_col: Column to calculate moving average for
            time_col: Time dimension column
            window: Window size (e.g., 3 for 3-period moving average)
        """
        result_df = data.copy()
        
        # Sort by time dimension
        result_df = self._sort_by_time_dimension(result_df, time_col)
        
        # Calculate moving average
        result_df[f'MA_{window}'] = result_df[metric_col].rolling(window=window, min_periods=1).mean()
        
        logger.info(f"✅ {window}-period moving average calculated")
        
        return result_df
    
    def _sort_by_time_dimension(self, data: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """
        Sort DataFrame by time dimension, handling fiscal year format
        
        Supports:
        - Fiscal year format (e.g., '2020-21', '2021-22')
        - Regular year format (e.g., 2020, 2021)
        - Date columns
        - Month names
        """
        result_df = data.copy()
        
        # Check if column contains fiscal year format (YYYY-YY)
        sample_values = result_df[time_col].astype(str).head(3).tolist()
        is_fiscal_year = any('-' in str(val) and len(str(val).split('-')) == 2 for val in sample_values)
        
        if is_fiscal_year:
            # Extract starting year from fiscal year format
            def extract_start_year(fy_string):
                try:
                    fy_str = str(fy_string).strip()
                    if '-' in fy_str:
                        start_year = fy_str.split('-')[0]
                        return int(start_year)
                    return 0
                except (ValueError, AttributeError, IndexError):
                    return 0
            
            result_df['_sort_key'] = result_df[time_col].apply(extract_start_year)
            result_df = result_df.sort_values('_sort_key').drop('_sort_key', axis=1)
            logger.info(f"🔄 Sorted by fiscal year: {time_col}")
        else:
            # Try regular sort (works for dates, years, month numbers)
            try:
                result_df = result_df.sort_values(time_col)
                logger.info(f"🔄 Sorted by {time_col}")
            except Exception as e:
                logger.warning(f"Could not sort by {time_col}: {e}")
        
        return result_df
    
    def calculate_per_unit(self, data: pd.DataFrame, metric_col: str, unit_col: str, 
                          dimension_col: str) -> pd.DataFrame:
        """
        Calculate per-unit metrics (e.g., revenue per order, sales per customer)
        
        Args:
            data: Raw transaction-level data
            metric_col: Column containing metric values (e.g., revenue)
            unit_col: Column to count units by (e.g., Invoice No for orders)
            dimension_col: Dimension to group by (e.g., State, Product)
            
        Returns:
            DataFrame with per-unit calculations
        """
        # Group by dimension
        grouped = data.groupby(dimension_col).agg({
            metric_col: 'sum',  # Total metric value
            unit_col: 'nunique'  # Count unique units (e.g., unique orders)
        }).reset_index()
        
        # Rename columns
        grouped.columns = [dimension_col, f'Total_{metric_col}', f'{unit_col}_Count']
        
        # Calculate per-unit value
        grouped[f'{metric_col}_per_{unit_col}'] = grouped[f'Total_{metric_col}'] / grouped[f'{unit_col}_Count']
        
        logger.info(f"✅ Per-unit calculation: {metric_col} per {unit_col} by {dimension_col}")
        
        return grouped
