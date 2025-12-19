import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataConnector:
    """Data connector to integrate with various data sources"""
    
    def __init__(self, data_folder='data', auto_load=True):
        """Initialize data connector with data folder path"""
        self.data_folder = data_folder
        self.cached_data = {}
        self.datasets = {}  # Store multiple datasets: {dataset_name: {table_name: dataframe}}
        self.current_dataset = None  # Track which dataset is currently active
        
        # Define sample aggregation keywords for detection
        self.aggregation_keywords = {
            'sum': ['sum', 'total', 'add', 'summation', 'sum of', 'total of'],
            'avg': ['average', 'avg', 'mean', 'average of', 'avg of'],
            'count': ['count', 'number of', 'how many', 'total count', 'count of'],
            'max': ['max', 'maximum', 'highest', 'largest', 'top value', 'max of'],
            'min': ['min', 'minimum', 'lowest', 'smallest', 'bottom value', 'min of']
        }
        
        # Load available Excel files only if auto_load is True
        if auto_load:
            self.load_all_datasets()
    
    def load_all_datasets(self):
        """Load all available datasets from subdirectories in data folder"""
        try:
            # Check if data folder exists
            if not os.path.exists(self.data_folder):
                print(f"❌ Data folder not found: {self.data_folder}")
                return
            
            # Look for subdirectories (each subdirectory is a dataset)
            for item in os.listdir(self.data_folder):
                item_path = os.path.join(self.data_folder, item)
                
                # Check if it's a directory
                if os.path.isdir(item_path):
                    dataset_name = item
                    self.load_dataset(dataset_name, item_path)
            
            # Also load Excel files from root data folder as "default" dataset
            excel_files = [f for f in os.listdir(self.data_folder) if f.endswith('.xlsx')]
            if excel_files:
                print(f"\n📁 Found {len(excel_files)} Excel files in root data folder:")
                self.load_dataset("default", self.data_folder, is_root=True)
            
            # Set the first dataset as current if available
            if self.datasets:
                self.current_dataset = list(self.datasets.keys())[0]
                self.switch_dataset(self.current_dataset)
                print(f"\n✅ Default dataset set to: {self.current_dataset}")
            else:
                print("\n⚠️  No datasets found!")
                    
        except Exception as e:
            print(f"❌ Error accessing data folder: {e}")
    
    def load_dataset(self, dataset_name, dataset_path, is_root=False):
        """Load a single dataset from a folder"""
        try:
            # Get only Excel files (not directories)
            items = os.listdir(dataset_path)
            excel_files = []
            
            for item in items:
                item_full_path = os.path.join(dataset_path, item)
                # Only include files, not directories
                if os.path.isfile(item_full_path) and item.endswith('.xlsx'):
                    excel_files.append(item)
            
            if not excel_files:
                if not is_root:  # Only warn if it's a dedicated dataset folder
                    print(f"   ⚠️  No Excel files in {dataset_name}")
                return
            
            print(f"\n📊 Loading dataset: {dataset_name}")
            print(f"   Found {len(excel_files)} Excel files:")
            
            self.datasets[dataset_name] = {}
            
            for file in excel_files:
                file_path = os.path.join(dataset_path, file)
                try:
                    # Try to load with first row as header
                    df = pd.read_excel(file_path, header=0)
                    
                    # Check if we got unnamed columns
                    unnamed_count = sum(1 for col in df.columns if str(col).startswith('Unnamed:'))
                    
                    # If most columns are unnamed, try reading with header in row 1 (0-indexed)
                    if unnamed_count > len(df.columns) * 0.5:  # If more than 50% unnamed
                        print(f"      ⚠️  Detected unnamed columns, trying alternate header row...")
                        try:
                            # Try header=1 (second row)
                            df_alt = pd.read_excel(file_path, header=1)
                            unnamed_alt = sum(1 for col in df_alt.columns if str(col).startswith('Unnamed:'))
                            
                            if unnamed_alt < unnamed_count:
                                df = df_alt
                                print(f"      ✅ Used row 2 as header")
                        except:
                            pass
                    
                    # If still mostly unnamed, it might be intentional or need manual fixing
                    # But we'll load it anyway
                    
                    table_name = file.replace('.xlsx', '').replace(' ', '_').lower()
                    self.datasets[dataset_name][table_name] = df
                    
                    # Show column info
                    if unnamed_count > 0:
                        print(f"      ⚠️  {file} -> {table_name} ({len(df)} rows, {unnamed_count} unnamed columns)")
                        print(f"         Please check if Excel file has proper headers!")
                    else:
                        print(f"      ✅ {file} -> {table_name} ({len(df)} rows, {len(df.columns)} columns)")
                    
                except Exception as e:
                    print(f"      ❌ Error loading {file}: {e}")
            
        except Exception as e:
            print(f"   ❌ Error loading dataset {dataset_name}: {e}")
    
    def switch_dataset(self, dataset_name):
        """Switch to a different dataset"""
        if dataset_name in self.datasets:
            self.current_dataset = dataset_name
            self.cached_data = self.datasets[dataset_name]
            print(f"✅ Switched to dataset: {dataset_name}")
            print(f"   Available tables: {list(self.cached_data.keys())}")
            return True
        else:
            print(f"❌ Dataset '{dataset_name}' not found")
            print(f"   Available datasets: {list(self.datasets.keys())}")
            return False
    
    def get_available_datasets(self):
        """Get list of all available datasets"""
        return list(self.datasets.keys())
    
    def get_current_dataset(self):
        """Get the currently active dataset name"""
        return self.current_dataset
    
    def get_dataset_info_all(self):
        """Get information about all datasets"""
        dataset_info = {}
        for dataset_name, tables in self.datasets.items():
            dataset_info[dataset_name] = {
                'name': dataset_name,
                'display_name': dataset_name.replace('_', ' ').title(),
                'table_count': len(tables),
                'tables': list(tables.keys())
            }
        return dataset_info
    
    def list_available_tables(self):
        """List all available data tables"""
        return list(self.cached_data.keys())
    
    def get_table_info(self, table_name):
        """Get information about a specific table"""
        if table_name in self.cached_data:
            df = self.cached_data[table_name]
            return {
                'name': table_name,
                'rows': len(df),
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
        return None
    
    def find_relevant_table(self, metric, dimension):
        """Find the most relevant table based on metric and dimension"""
        # Simple heuristic to find relevant tables
        metric_lower = str(metric).lower()
        dimension_lower = str(dimension).lower()
        
        # Check each table for relevant columns
        best_match = None
        best_score = 0
        
        for table_name, df in self.cached_data.items():
            score = 0
            columns_lower = [col.lower() for col in df.columns]
            
            # Check for metric-related columns
            for col in columns_lower:
                if any(keyword in col for keyword in [metric_lower, 'sales', 'revenue', 'profit', 'amount']):
                    score += 3
                if any(keyword in col for keyword in [dimension_lower, 'region', 'product', 'category']):
                    score += 2
            
            # Prefer fact tables (usually contain sales data)
            if 'fact' in table_name or 'sales' in table_name:
                score += 1
                
            if score > best_score:
                best_score = score
                best_match = table_name
        
        return best_match, best_score
    
    def extract_all_columns_info(self):
        """Extract all column information from all available tables"""
        all_columns_info = {}
        
        for table_name, df in self.cached_data.items():
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            all_columns_info[table_name] = {
                'all_columns': list(df.columns),
                'numeric_columns': numeric_cols,
                'text_columns': text_cols,
                'date_columns': date_cols,
                'row_count': len(df)
            }
        
        return all_columns_info

    def sort_fiscal_year_column(self, df, column_name):
        """Sort dataframe by fiscal year column (handles formats like '2024-25', '2023-24', etc.)"""
        try:
            # Check if the column contains fiscal year format (YYYY-YY)
            sample_values = df[column_name].astype(str).head(5).tolist()
            is_fiscal_year = any('-' in str(val) and len(str(val).split('-')) == 2 for val in sample_values)
            
            if is_fiscal_year:
                # Extract the starting year from fiscal year format
                def extract_start_year(fy_string):
                    try:
                        fy_str = str(fy_string).strip()
                        if '-' in fy_str:
                            start_year = fy_str.split('-')[0]
                            return int(start_year)
                        return 0
                    except:
                        return 0
                
                # Create a temporary sorting column
                df['_sort_key'] = df[column_name].apply(extract_start_year)
                df = df.sort_values('_sort_key').drop('_sort_key', axis=1)
                print(f"🔄 Sorted by fiscal year: {column_name}")
            
        except Exception as e:
            print(f"⚠️ Could not sort fiscal year column: {e}")
        
        return df

    def find_columns_mentioned_in_query(self, query_text, table_name):
        """Find column names that are mentioned in the query"""
        if table_name not in self.cached_data:
            return [], []
        
        df = self.cached_data[table_name]
        query_lower = str(query_text).lower()
        
        mentioned_columns = []
        column_matches = {}
        
        # Check each column name against the query
        for col in df.columns:
            # Strip spaces from column name for matching
            col_stripped = col.strip()
            col_lower = col_stripped.lower()
            col_words = col_lower.split()
            
            # Direct column name match (with stripped version)
            if col_lower in query_lower:
                mentioned_columns.append(col)
                column_matches[col] = 'exact_match'
                continue
            
            # Partial word matches
            for word in col_words:
                if len(word) > 2 and word in query_lower:
                    mentioned_columns.append(col)
                    column_matches[col] = 'partial_match'
                    break
            
            # Check for common variations
            variations = self.get_column_variations(col_lower)
            for variation in variations:
                if variation in query_lower:
                    mentioned_columns.append(col)
                    column_matches[col] = 'variation_match'
                    break
        
        return list(set(mentioned_columns)), column_matches

    def get_column_variations(self, column_name):
        """Get common variations of column names"""
        variations = [column_name]
        
        # Common replacements
        replacements = {
            'key': ['id', 'identifier', 'code'],
            'amt': ['amount', 'value'],
            'qty': ['quantity', 'count'],
            'desc': ['description', 'name'],
            'cat': ['category', 'type'],
            'prod': ['product'],
            'cust': ['customer', 'client'],
            'terr': ['territory', 'region'],
            'sales': ['sale', 'revenue', 'income'],
            'cost': ['price', 'expense'],
            'date': ['time', 'day', 'month', 'year']
        }
        
        for abbrev, full_words in replacements.items():
            if abbrev in column_name:
                for word in full_words:
                    variations.append(column_name.replace(abbrev, word))
            for word in full_words:
                if word in column_name:
                    variations.append(column_name.replace(word, abbrev))
        
        return variations
    
    def detect_aggregation_from_query(self, query_text):
        """Detect aggregation function from query text"""
        if not query_text:
            return None
        
        query_lower = query_text.lower()
        
        # Check for aggregation keywords in query
        for agg_func, keywords in self.aggregation_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    print(f"🔍 Detected aggregation: {agg_func} (keyword: '{keyword}')")
                    return agg_func
        
        # Default to sum if no specific aggregation detected
        return None

    def get_data_for_chart(self, metric, dimension, filters=None, aggregation=None, query_text=""):
        """Get actual data for chart generation with intelligent column matching"""
        # Find relevant table
        table_name, score = self.find_relevant_table(metric, dimension)
        
        if not table_name or score == 0:
            raise Exception(f"❌ No relevant data table found for metric: {metric}, dimension: {dimension}")
        
        df = self.cached_data[table_name].copy()
        print(f"📊 Using table: {table_name} (relevance score: {score})")
        
        # Extract all column information
        columns_info = self.extract_all_columns_info()
        print(f"📋 Available columns in {table_name}: {columns_info[table_name]['all_columns']}")
        
        # Find columns mentioned in query
        if query_text:
            mentioned_cols, matches = self.find_columns_mentioned_in_query(query_text, table_name)
            if mentioned_cols:
                print(f"🔍 Columns mentioned in query: {mentioned_cols}")
                print(f"🎯 Match types: {matches}")
        else:
            mentioned_cols, matches = [], {}
        
    def get_data_for_chart(self, metric, dimension, filters=None, aggregation=None, query_text=""):
        """Get actual data for chart generation with intelligent column matching"""
        # Find relevant table
        table_name, score = self.find_relevant_table(metric, dimension)
        
        if not table_name or score == 0:
            raise Exception(f"❌ No relevant data table found for metric: {metric}, dimension: {dimension}")
        
        df = self.cached_data[table_name].copy()
        print(f"📊 Using table: {table_name} (relevance score: {score})")
        
        # Extract all column information
        columns_info = self.extract_all_columns_info()
        print(f"📋 Available columns in {table_name}: {columns_info[table_name]['all_columns']}")
        
        # Find columns mentioned in query
        if query_text:
            mentioned_cols, matches = self.find_columns_mentioned_in_query(query_text, table_name)
            if mentioned_cols:
                print(f"🔍 Columns mentioned in query: {mentioned_cols}")
                print(f"🎯 Match types: {matches}")
                
                # Smart logic: if we have mentioned columns, try to use them intelligently
                # If metric/dimension entities don't match well, use mentioned columns
                if len(mentioned_cols) >= 2:
                    # Try to assign mentioned columns to metric and dimension based on data types
                    numeric_mentioned = [col for col in mentioned_cols if col in df.select_dtypes(include=['number']).columns]
                    
                    if len(numeric_mentioned) >= 2:
                        # Both can be metrics, use query structure to decide
                        query_lower = query_text.lower()
                        words = query_lower.split()
                        
                        # Find "by" keyword to determine metric vs dimension
                        if 'by' in words:
                            by_index = words.index('by')
                            metric_part = ' '.join(words[:by_index])
                            dimension_part = ' '.join(words[by_index+1:])
                            
                            # Find which mentioned column belongs to which part
                            metric_col = None
                            dimension_col = None
                            
                            for col in mentioned_cols:
                                if col.lower() in metric_part:
                                    metric_col = col
                                elif col.lower() in dimension_part:
                                    dimension_col = col
                            
                            if metric_col and dimension_col:
                                print(f"   🎯 Smart assignment: {metric_col} (metric), {dimension_col} (dimension)")
                                result_df = df[[dimension_col, metric_col]].copy()
                                result_df.columns = [str(dimension).replace(' ', '_'), str(metric).replace(' ', '_')]
                                
                                # Apply filters and aggregation
                                if filters:
                                    result_df = self.apply_filters(result_df, df, filters)
                                if aggregation:
                                    result_df = self.apply_aggregation(result_df, aggregation)
                                
                                result_df = result_df.dropna()
                                if result_df.empty:
                                    raise Exception(f"❌ No data remaining after applying filters and aggregation")
                                return result_df
    def get_data_for_chart_column_based(self, metric, dimension, filters=None, aggregation=None, query_text=""):
        """Get actual data for chart generation prioritizing column names over NER entities"""
        # Find relevant table
        table_name, score = self.find_relevant_table(metric, dimension)
        
        if not table_name or score == 0:
            # Try to find any table with mentioned columns
            if query_text:
                for tname, df in self.cached_data.items():
                    mentioned_cols, _ = self.find_columns_mentioned_in_query(query_text, tname)
                    if len(mentioned_cols) >= 2:
                        table_name = tname
                        print(f"📊 Using table: {table_name} (found mentioned columns)")
                        break
            
            if not table_name:
                raise Exception(f"❌ No relevant data table found for metric: {metric}, dimension: {dimension}")
        else:
            print(f"📊 Using table: {table_name} (relevance score: {score})")
        
        df = self.cached_data[table_name].copy()
        
        # Extract all column information
        columns_info = self.extract_all_columns_info()
        print(f"📋 Available columns in {table_name}: {columns_info[table_name]['all_columns']}")
        
        # Detect aggregation from query if not explicitly provided
        if not aggregation and query_text:
            aggregation = self.detect_aggregation_from_query(query_text)
            if aggregation:
                print(f"✅ Using detected aggregation: {aggregation}")
        
        # Find columns mentioned in query - THIS IS NOW THE PRIMARY METHOD
        mentioned_cols, matches = self.find_columns_mentioned_in_query(query_text, table_name)
        
        if mentioned_cols:
            print(f"🔍 Columns mentioned in query: {mentioned_cols}")
            print(f"🎯 Match types: {matches}")
            
            # COLUMN-BASED APPROACH: Use mentioned columns directly
            if len(mentioned_cols) >= 2:
                # Try to assign mentioned columns intelligently based on query structure
                metric_col, dimension_col = self.assign_columns_from_query(mentioned_cols, query_text, df)
                
                if metric_col and dimension_col:
                    print(f"   🎯 Column-based assignment: {metric_col} (metric), {dimension_col} (dimension)")
                    result_df = df[[dimension_col, metric_col]].copy()
                    result_df.columns = [f'dimension_{dimension_col}', f'metric_{metric_col}']
                    
                    # Apply aggregation FIRST (group by dimension)
                    if aggregation:
                        print(f"📊 Applying aggregation: {aggregation}")
                        result_df = self.apply_aggregation(result_df, aggregation)
                    else:
                        # Default aggregation for charts (sum)
                        print(f"📊 Applying default aggregation: sum")
                        result_df = result_df.groupby(f'dimension_{dimension_col}')[f'metric_{metric_col}'].sum().reset_index()
                    
                    # Sort by fiscal year if dimension is FY-like
                    result_df = self.sort_fiscal_year_column(result_df, f'dimension_{dimension_col}')
                    
                    # Apply filters AFTER aggregation (especially important for "top N" filters)
                    if filters:
                        result_df = self.apply_filters(result_df, df, filters)
                    
                    result_df = result_df.dropna()
                    if result_df.empty:
                        raise Exception(f"❌ No data remaining after applying filters and aggregation")
                    # Debugging output
                    try:
                        print("🔎 get_data_for_chart_column_based - result_df (column-based, >=2):")
                        print(result_df.head(10).to_dict(orient='records'))
                        print("🔎 dtypes:", result_df.dtypes.to_dict())
                    except Exception:
                        print("🔎 get_data_for_chart_column_based - could not print result_df")
                    return result_df
            
            elif len(mentioned_cols) == 1:
                # If only one column mentioned, use it as metric and find a suitable dimension
                mentioned_col = mentioned_cols[0]
                if mentioned_col in df.select_dtypes(include=['number']).columns:
                    # Use as metric, find categorical dimension
                    text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if text_cols:
                        dimension_col = text_cols[0]
                        metric_col = mentioned_col
                        print(f"   🎯 Single column approach: {metric_col} (metric), {dimension_col} (dimension)")
                    else:
                        # Use another numeric column as dimension
                        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col != mentioned_col]
                        if numeric_cols:
                            dimension_col = numeric_cols[0]
                            metric_col = mentioned_col
                            print(f"   🎯 Numeric-numeric approach: {metric_col} (metric), {dimension_col} (dimension)")
                        else:
                            raise Exception(f"❌ Cannot find suitable dimension for {mentioned_col}")
                else:
                    # Use as dimension, find numeric metric
                    dimension_col = mentioned_col
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        metric_col = numeric_cols[0]
                        print(f"   🎯 Single column approach: {metric_col} (metric), {dimension_col} (dimension)")
                    else:
                        raise Exception(f"❌ No numeric columns found for metric")
                
                result_df = df[[dimension_col, metric_col]].copy()
                result_df.columns = [f'dimension_{dimension_col}', f'metric_{metric_col}']
                
                # Apply aggregation
                if aggregation:
                    print(f"📊 Applying aggregation: {aggregation}")
                    result_df = self.apply_aggregation(result_df, aggregation)
                else:
                    # Default aggregation for charts (sum)
                    print(f"📊 Applying default aggregation: sum")
                    result_df = result_df.groupby(f'dimension_{dimension_col}')[f'metric_{metric_col}'].sum().reset_index()
                
                # Sort by fiscal year if dimension is FY-like
                result_df = self.sort_fiscal_year_column(result_df, f'dimension_{dimension_col}')
                
                # Apply filters
                if filters:
                    result_df = self.apply_filters(result_df, df, filters)
                
                result_df = result_df.dropna()
                if result_df.empty:
                    raise Exception(f"❌ No data remaining after applying filters and aggregation")
                # Debugging output
                try:
                    print("🔎 get_data_for_chart_column_based - result_df (single-column case):")
                    print(result_df.head(10).to_dict(orient='records'))
                    print("🔎 dtypes:", result_df.dtypes.to_dict())
                except Exception:
                    print("🔎 get_data_for_chart_column_based - could not print result_df")
                return result_df
        
        # Fallback to original NER-based approach if no columns mentioned
        print("⚠️ No columns mentioned in query, falling back to NER-based approach")
        return self.get_data_for_chart_ner_based(metric, dimension, filters, aggregation, query_text, df, table_name)

    def assign_columns_from_query(self, mentioned_cols, query_text, df):
        """Assign mentioned columns to metric and dimension based on query structure"""
        query_lower = query_text.lower()
        words = query_lower.split()
        
        # Method 1: Use "by" keyword to determine assignment
        if 'by' in words:
            by_index = words.index('by')
            metric_part = ' '.join(words[:by_index])
            dimension_part = ' '.join(words[by_index+1:])
            
            metric_col = None
            dimension_col = None
            
            # Find which mentioned column belongs to which part
            for col in mentioned_cols:
                if col.lower() in metric_part:
                    metric_col = col
                elif col.lower() in dimension_part:
                    dimension_col = col
            
            if metric_col and dimension_col:
                return metric_col, dimension_col
        
        # Method 2: Use data types to make intelligent assignment
        numeric_mentioned = [col for col in mentioned_cols if col in df.select_dtypes(include=['number']).columns]
        text_mentioned = [col for col in mentioned_cols if col in df.select_dtypes(include=['object', 'category']).columns]
        
        if len(numeric_mentioned) >= 1 and len(text_mentioned) >= 1:
            # One numeric (metric) and one categorical (dimension)
            return numeric_mentioned[0], text_mentioned[0]
        elif len(numeric_mentioned) >= 2:
            # Both numeric - use first as metric, second as dimension
            return numeric_mentioned[0], numeric_mentioned[1]
        elif len(text_mentioned) >= 2:
            # Both categorical - unusual but handle it
            # Find if any has numeric values we can use as metric
            for col in text_mentioned:
                try:
                    pd.to_numeric(df[col])
                    # This column can be converted to numeric
                    other_cols = [c for c in text_mentioned if c != col]
                    if other_cols:
                        return col, other_cols[0]
                except:
                    continue
        
        # Method 3: Use order in query
        if len(mentioned_cols) >= 2:
            # Find order of appearance in query
            col_positions = []
            for col in mentioned_cols:
                pos = query_lower.find(col.lower())
                if pos != -1:
                    col_positions.append((pos, col))
            
            if len(col_positions) >= 2:
                col_positions.sort()  # Sort by position
                return col_positions[0][1], col_positions[1][1]  # First as metric, second as dimension
        
        return None, None

    def get_data_for_chart_ner_based(self, metric, dimension, filters, aggregation, query_text, df, table_name):
        """Original NER-based approach as fallback"""
        mentioned_cols, matches = self.find_columns_mentioned_in_query(query_text, table_name)
        
        # Detect aggregation from query if not explicitly provided
        if not aggregation and query_text:
            aggregation = self.detect_aggregation_from_query(query_text)
            if aggregation:
                print(f"✅ Using detected aggregation: {aggregation}")
        
        # Try to find matching columns with priority to mentioned columns
        metric_col = self.find_best_column_match(df, metric, 'metric', mentioned_cols, matches)
        dimension_col = self.find_best_column_match(df, dimension, 'dimension', mentioned_cols, matches)
        
        if not metric_col and not dimension_col:
            # If no specific matches, try to use any numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and text_cols:
                metric_col = numeric_cols[0]
                dimension_col = text_cols[0]
                print(f"   📋 Using fallback columns: {metric_col} (metric), {dimension_col} (dimension)")
            else:
                raise Exception(f"❌ No suitable columns found in table {table_name}")
        elif not metric_col:
            # Find any numeric column for metric
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                metric_col = numeric_cols[0]
                print(f"   📋 Using fallback metric column: {metric_col}")
            else:
                raise Exception(f"❌ No numeric columns found for metric in table {table_name}")
        elif not dimension_col:
            # Find any categorical column for dimension
            text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if text_cols:
                dimension_col = text_cols[0]
                print(f"   📋 Using fallback dimension column: {dimension_col}")
            else:
                raise Exception(f"❌ No categorical columns found for dimension in table {table_name}")
        
        print(f"   ✅ Selected metric column: {metric_col}")
        print(f"   ✅ Selected dimension column: {dimension_col}")
        
        # Create result dataframe
        result_df = df[[dimension_col, metric_col]].copy()
        result_df.columns = [str(dimension).replace(' ', '_'), str(metric).replace(' ', '_')]
        
        # Apply aggregation FIRST
        if aggregation:
            print(f"📊 Applying aggregation: {aggregation}")
            result_df = self.apply_aggregation(result_df, aggregation)
        else:
            # Default aggregation for charts (sum)
            print(f"📊 Applying default aggregation: sum")
            dimension_col_renamed = result_df.columns[0]
            metric_col_renamed = result_df.columns[1]
            result_df = result_df.groupby(dimension_col_renamed)[metric_col_renamed].sum().reset_index()
        
        # Sort by fiscal year if dimension is FY-like
        dimension_col_renamed = result_df.columns[0]
        result_df = self.sort_fiscal_year_column(result_df, dimension_col_renamed)
        
        # Apply filters AFTER aggregation
        if filters:
            result_df = self.apply_filters(result_df, df, filters)
        
        # Remove any null values
        result_df = result_df.dropna()
        
        if result_df.empty:
            raise Exception(f"❌ No data remaining after applying filters and aggregation")
        # Debugging output
        try:
            print("🔎 get_data_for_chart_column_based - result_df (ner-fallback):")
            print(result_df.head(10).to_dict(orient='records'))
            print("🔎 dtypes:", result_df.dtypes.to_dict())
        except Exception:
            print("🔎 get_data_for_chart_column_based - could not print result_df")

        return result_df
    
    def find_column_match(self, df, target, alternatives):
        """Find the best matching column name - IMPROVED MATCHING"""
        target_lower = str(target).lower()
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Exact match
        if target_lower in columns_lower:
            return columns_lower[target_lower]
        
        # Partial match with target - check if target is contained in column name
        for col_lower, col_original in columns_lower.items():
            if target_lower in col_lower:
                return col_original
        
        # Check if column name is contained in target
        for col_lower, col_original in columns_lower.items():
            if col_lower in target_lower:
                return col_original
        
        # Match with alternatives - broader search
        for alt in alternatives:
            alt_lower = alt.lower()
            for col_lower, col_original in columns_lower.items():
                if alt_lower in col_lower or col_lower in alt_lower:
                    return col_original
        
        # Last resort - find any numeric column for metrics
        if target_lower in ['sales', 'revenue', 'profit', 'amount', 'value', 'price', 'cost']:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                return numeric_cols[0]  # Return first numeric column
        
        # Last resort - find any text/categorical column for dimensions
        if target_lower in ['region', 'category', 'product', 'customer', 'territory']:
            text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if text_cols:
                return text_cols[0]  # Return first text column
        
    def find_best_column_match(self, df, target, target_type, mentioned_cols, matches):
        """Find the best matching column with priority to mentioned columns"""
        target_lower = str(target).lower()
        
        # Priority 1: Check if target exactly matches any mentioned column
        for col in mentioned_cols:
            col_lower = col.lower()
            if target_lower == col_lower or target_lower.replace(' ', '') == col_lower.replace(' ', ''):
                if target_type == 'metric' and col in df.select_dtypes(include=['number']).columns:
                    print(f"   🎯 Found exact mentioned metric column: {col}")
                    return col
                elif target_type == 'dimension':
                    print(f"   🎯 Found exact mentioned dimension column: {col}")
                    return col
        
        # Priority 2: Check if any mentioned column contains the target or vice versa
        for col in mentioned_cols:
            col_lower = col.lower()
            if target_lower in col_lower or col_lower in target_lower:
                if target_type == 'metric' and col in df.select_dtypes(include=['number']).columns:
                    print(f"   🎯 Found mentioned metric column (partial match): {col}")
                    return col
                elif target_type == 'dimension':
                    print(f"   🎯 Found mentioned dimension column (partial match): {col}")
                    return col
        
        # Priority 3: Check if target is similar to any mentioned column (handle typos)
        for col in mentioned_cols:
            col_lower = col.lower()
            # Handle common typos and variations
            if self.is_similar_column(target_lower, col_lower):
                if target_type == 'metric' and col in df.select_dtypes(include=['number']).columns:
                    print(f"   🎯 Found similar mentioned metric column: {col} (for {target})")
                    return col
                elif target_type == 'dimension':
                    print(f"   🎯 Found similar mentioned dimension column: {col} (for {target})")
                    return col
        
        # Priority 4: Use original column matching logic
        if target_type == 'metric':
            alternatives = ['sales', 'revenue', 'profit', 'amount', 'value', 'price', 'cost', 'total', 'freight', 'tax']
        else:
            alternatives = ['region', 'product', 'category', 'territory', 'customer', 'type', 'name', 'key']
        
        matched_col = self.find_column_match(df, target, alternatives)
        if matched_col:
            print(f"   🔍 Found {target_type} column via matching: {matched_col}")
            return matched_col
        
        # Priority 5: If target mentioned in query but not found above, try any mentioned column of right type
        for col in mentioned_cols:
            if target_type == 'metric' and col in df.select_dtypes(include=['number']).columns:
                print(f"   🔄 Using mentioned numeric column for metric: {col}")
                return col
            elif target_type == 'dimension' and col in df.select_dtypes(include=['object', 'category']).columns:
                print(f"   🔄 Using mentioned categorical column for dimension: {col}")
                return col
        
        return None

    def is_similar_column(self, target, column):
        """Check if target and column are similar (handle typos)"""
        # Common typos and variations
        variations = {
            'fright': 'freight',
            'taxamt': 'taxamt',
            'unitprice': 'unitprice',
            'orderquantity': 'orderquantity',
            'productkey': 'productkey',
            'customerkey': 'customerkey'
        }
        
        target_clean = target.replace(' ', '').replace('_', '')
        column_clean = column.replace(' ', '').replace('_', '')
        
        # Check if they're variations of each other
        if target_clean in variations and variations[target_clean] == column_clean:
            return True
        if column_clean in variations and variations[column_clean] == target_clean:
            return True
        
        # Check if they're very similar (simple edit distance)
        if abs(len(target_clean) - len(column_clean)) <= 2:
            if target_clean in column_clean or column_clean in target_clean:
                return True
        
        return False

    def apply_filters(self, result_df, original_df, filters):
        """Apply filters to the data"""
        for filter_item in filters:
            filter_lower = filter_item.lower()
            
            # Handle "top N" filters
            if 'top' in filter_lower:
                import re
                match = re.search(r'(\d+)', filter_item)
                if match:
                    top_n = int(match.group(1))
                    metric_col = result_df.columns[1]  # Second column is metric
                    result_df = result_df.nlargest(top_n, metric_col)
            
            # Handle year filters
            elif any(year in filter_lower for year in ['2024', '2023', '2022']):
                # Look for date columns in original data
                for col in original_df.columns:
                    if 'date' in col.lower() or 'year' in col.lower():
                        try:
                            year = next(year for year in ['2024', '2023', '2022'] if year in filter_lower)
                            if col in original_df.columns:
                                year_mask = original_df[col].astype(str).str.contains(year, na=False)
                                filtered_indices = original_df[year_mask].index
                                result_df = result_df.loc[result_df.index.intersection(filtered_indices)]
                                break
                        except:
                            pass
            
            # Handle region filters
            elif filter_lower in ['north', 'south', 'east', 'west']:
                dimension_col = result_df.columns[0]  # First column is dimension
                region_mask = result_df[dimension_col].astype(str).str.contains(filter_item, case=False, na=False)
                result_df = result_df[region_mask]
        
        return result_df
    
    def apply_aggregation(self, df, aggregation):
        """Apply aggregation to the data"""
        dimension_col = df.columns[0]
        metric_col = df.columns[1]
        
        if aggregation in ['sum', 'total']:
            result = df.groupby(dimension_col)[metric_col].sum().reset_index()
        elif aggregation in ['avg', 'average', 'mean']:
            result = df.groupby(dimension_col)[metric_col].mean().reset_index()
        elif aggregation == 'count':
            result = df.groupby(dimension_col)[metric_col].count().reset_index()
        elif aggregation == 'max':
            result = df.groupby(dimension_col)[metric_col].max().reset_index()
        elif aggregation == 'min':
            result = df.groupby(dimension_col)[metric_col].min().reset_index()
        else:
            result = df  # No aggregation
        
        return result

    def display_data_info(self):
        """Display information about available data sources"""
        print("\n📊 Available Data Sources:")
        print("=" * 50)
        
        for table_name, df in self.cached_data.items():
            print(f"\n📋 Table: {table_name}")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Show data types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols:
                print(f"   Numeric columns: {numeric_cols}")
            if text_cols:
                print(f"   Text columns: {text_cols}")
        
        print("\n" + "=" * 50)