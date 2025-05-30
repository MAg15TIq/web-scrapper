"""
Distributed result aggregation system using Apache Arrow.
"""
import time
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    from pyarrow import Table
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

class ResultAggregator:
    """
    Distributed result aggregation system using Apache Arrow.
    """
    def __init__(self):
        """Initialize the result aggregator."""
        self.logger = logging.getLogger("result_aggregator")
        
        if not ARROW_AVAILABLE:
            self.logger.warning("PyArrow is not available. Using pandas for aggregation instead.")
    
    def merge_results(self, chunks: List[Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]]) -> pd.DataFrame:
        """
        Merge multiple result chunks into a single DataFrame.
        
        Args:
            chunks: List of result chunks. Each chunk can be a dictionary, list of dictionaries, or DataFrame.
            
        Returns:
            A pandas DataFrame containing the merged results.
        """
        if not chunks:
            return pd.DataFrame()
        
        start_time = time.time()
        self.logger.info(f"Merging {len(chunks)} result chunks")
        
        # Convert all chunks to DataFrames
        dataframes = []
        for chunk in chunks:
            if isinstance(chunk, pd.DataFrame):
                dataframes.append(chunk)
            elif isinstance(chunk, dict):
                dataframes.append(pd.DataFrame([chunk]))
            elif isinstance(chunk, list):
                dataframes.append(pd.DataFrame(chunk))
            else:
                self.logger.warning(f"Unsupported chunk type: {type(chunk)}")
        
        if not dataframes:
            return pd.DataFrame()
        
        # Use Arrow for merging if available
        if ARROW_AVAILABLE:
            try:
                # Convert DataFrames to Arrow Tables
                tables = [Table.from_pandas(df) for df in dataframes]
                
                # Concatenate tables
                merged_table = pa.concat_tables(tables)
                
                # Convert back to pandas
                result = merged_table.to_pandas()
                
                self.logger.info(f"Merged {len(chunks)} chunks using Arrow in {time.time() - start_time:.2f} seconds")
                return result
            except Exception as e:
                self.logger.error(f"Error merging with Arrow: {str(e)}")
                self.logger.info("Falling back to pandas for merging")
        
        # Fallback to pandas concat
        result = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Merged {len(chunks)} chunks using pandas in {time.time() - start_time:.2f} seconds")
        return result
    
    def resolve_conflicts(self, df: pd.DataFrame, id_columns: List[str], timestamp_column: Optional[str] = None,
                         source_column: Optional[str] = None, source_authority: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Resolve conflicts in the merged results.
        
        Args:
            df: DataFrame containing the merged results.
            id_columns: List of column names that uniquely identify a record.
            timestamp_column: Optional column name containing timestamps for versioning.
            source_column: Optional column name containing the source of each record.
            source_authority: Optional dictionary mapping sources to authority scores.
            
        Returns:
            A DataFrame with conflicts resolved.
        """
        if df.empty:
            return df
        
        start_time = time.time()
        self.logger.info(f"Resolving conflicts in {len(df)} records")
        
        # Create a composite key from ID columns
        if len(id_columns) == 1:
            df["_composite_key"] = df[id_columns[0]].astype(str)
        else:
            df["_composite_key"] = df[id_columns].astype(str).agg("-".join, axis=1)
        
        # Check for duplicates
        duplicate_keys = df["_composite_key"].duplicated(keep=False)
        if not duplicate_keys.any():
            # No conflicts to resolve
            df = df.drop(columns=["_composite_key"])
            self.logger.info(f"No conflicts found in {len(df)} records")
            return df
        
        # Get duplicated records
        duplicates = df[duplicate_keys].copy()
        non_duplicates = df[~duplicate_keys].copy()
        
        self.logger.info(f"Found {len(duplicates)} records with conflicts")
        
        # Resolve conflicts
        resolved = []
        
        # Group by composite key
        for key, group in duplicates.groupby("_composite_key"):
            if len(group) == 1:
                # No conflict for this key
                resolved.append(group)
                continue
            
            # Apply resolution strategies
            if timestamp_column and timestamp_column in group.columns:
                # Use the most recent record
                try:
                    latest = group.loc[group[timestamp_column].idxmax()]
                    resolved.append(pd.DataFrame([latest]))
                    continue
                except Exception as e:
                    self.logger.warning(f"Error resolving by timestamp: {str(e)}")
            
            if source_column and source_column in group.columns and source_authority:
                # Use the record from the most authoritative source
                try:
                    group["_authority"] = group[source_column].map(
                        lambda s: source_authority.get(s, 0)
                    )
                    most_authoritative = group.loc[group["_authority"].idxmax()]
                    resolved.append(pd.DataFrame([most_authoritative]))
                    continue
                except Exception as e:
                    self.logger.warning(f"Error resolving by source authority: {str(e)}")
            
            # Fallback: merge all values
            merged_record = self._merge_record_values(group)
            resolved.append(pd.DataFrame([merged_record]))
        
        # Combine resolved records with non-duplicates
        if resolved:
            resolved_df = pd.concat(resolved, ignore_index=True)
            result = pd.concat([non_duplicates, resolved_df], ignore_index=True)
        else:
            result = non_duplicates
        
        # Remove temporary columns
        for col in ["_composite_key", "_authority"]:
            if col in result.columns:
                result = result.drop(columns=[col])
        
        self.logger.info(f"Resolved conflicts in {time.time() - start_time:.2f} seconds")
        return result
    
    def _merge_record_values(self, group: pd.DataFrame) -> Dict[str, Any]:
        """
        Merge values from multiple records into a single record.
        
        Args:
            group: DataFrame containing records with the same ID.
            
        Returns:
            A dictionary representing the merged record.
        """
        merged = {}
        
        # Use the first record as a base
        base_record = group.iloc[0].to_dict()
        merged.update(base_record)
        
        # For each column, merge values if they differ
        for col in group.columns:
            if col in ["_composite_key", "_authority"]:
                continue
            
            values = group[col].dropna().unique()
            if len(values) <= 1:
                # All values are the same or null, use the base value
                continue
            
            # For list columns, combine all unique values
            if isinstance(base_record[col], list):
                all_items = []
                for val in group[col]:
                    if isinstance(val, list):
                        all_items.extend(val)
                    else:
                        all_items.append(val)
                merged[col] = list(set(all_items))
            
            # For string columns, use the longest value
            elif all(isinstance(v, str) for v in values):
                merged[col] = max(values, key=len)
            
            # For numeric columns, use the average
            elif all(isinstance(v, (int, float)) for v in values):
                merged[col] = float(np.mean(values))
            
            # Default: keep the base value
        
        return merged
    
    def calculate_similarity(self, record1: Dict[str, Any], record2: Dict[str, Any], 
                            text_columns: Optional[List[str]] = None) -> float:
        """
        Calculate semantic similarity between two records.
        
        Args:
            record1: First record.
            record2: Second record.
            text_columns: Optional list of columns to use for text similarity.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Simple field-by-field comparison
        common_keys = set(record1.keys()) & set(record2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        total = len(common_keys)
        
        for key in common_keys:
            val1 = record1[key]
            val2 = record2[key]
            
            # Skip None values
            if val1 is None or val2 is None:
                total -= 1
                continue
            
            # Exact match
            if val1 == val2:
                matches += 1
                continue
            
            # String similarity for text columns
            if text_columns and key in text_columns and isinstance(val1, str) and isinstance(val2, str):
                # Simple Jaccard similarity for strings
                tokens1 = set(val1.lower().split())
                tokens2 = set(val2.lower().split())
                
                if tokens1 and tokens2:
                    intersection = len(tokens1 & tokens2)
                    union = len(tokens1 | tokens2)
                    similarity = intersection / union if union > 0 else 0
                    matches += similarity
                    continue
            
            # Numeric similarity
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity = 1 - min(1, abs(val1 - val2) / max_val)
                    matches += similarity
                    continue
        
        return matches / total if total > 0 else 0.0
    
    def deduplicate_by_similarity(self, df: pd.DataFrame, threshold: float = 0.8, 
                                 text_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Deduplicate records based on semantic similarity.
        
        Args:
            df: DataFrame to deduplicate.
            threshold: Similarity threshold for considering records as duplicates.
            text_columns: Optional list of columns to use for text similarity.
            
        Returns:
            Deduplicated DataFrame.
        """
        if df.empty or len(df) == 1:
            return df
        
        start_time = time.time()
        self.logger.info(f"Deduplicating {len(df)} records by similarity")
        
        # Convert to records for easier processing
        records = df.to_dict(orient="records")
        unique_records = []
        duplicate_indices = set()
        
        # O(n²) comparison - can be optimized for large datasets
        for i, record1 in enumerate(records):
            if i in duplicate_indices:
                continue
            
            unique_records.append(record1)
            
            for j in range(i + 1, len(records)):
                if j in duplicate_indices:
                    continue
                
                record2 = records[j]
                similarity = self.calculate_similarity(record1, record2, text_columns)
                
                if similarity >= threshold:
                    duplicate_indices.add(j)
        
        result = pd.DataFrame(unique_records)
        self.logger.info(f"Removed {len(duplicate_indices)} duplicate records by similarity in {time.time() - start_time:.2f} seconds")
        return result
    
    def aggregate_results(self, chunks: List[Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]],
                         id_columns: List[str], timestamp_column: Optional[str] = None,
                         source_column: Optional[str] = None, source_authority: Optional[Dict[str, float]] = None,
                         similarity_threshold: Optional[float] = None, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate results from multiple chunks with conflict resolution.
        
        Args:
            chunks: List of result chunks.
            id_columns: List of column names that uniquely identify a record.
            timestamp_column: Optional column name containing timestamps for versioning.
            source_column: Optional column name containing the source of each record.
            source_authority: Optional dictionary mapping sources to authority scores.
            similarity_threshold: Optional threshold for similarity-based deduplication.
            text_columns: Optional list of columns to use for text similarity.
            
        Returns:
            A pandas DataFrame containing the aggregated results.
        """
        # Merge chunks
        merged = self.merge_results(chunks)
        if merged.empty:
            return merged
        
        # Resolve conflicts
        if id_columns:
            resolved = self.resolve_conflicts(
                merged, id_columns, timestamp_column, source_column, source_authority
            )
        else:
            resolved = merged
        
        # Deduplicate by similarity if requested
        if similarity_threshold is not None and similarity_threshold > 0:
            deduplicated = self.deduplicate_by_similarity(
                resolved, similarity_threshold, text_columns
            )
            return deduplicated
        
        return resolved

# Create a singleton instance
result_aggregator = ResultAggregator()
