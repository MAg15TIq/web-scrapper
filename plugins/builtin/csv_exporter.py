"""
CSV Exporter Plugin
Exports scraped data to CSV format with advanced formatting options.
"""

import csv
import io
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from plugins.base_plugin import BasePlugin, PluginMetadata, PluginType, PluginResult


class CSVExporterPlugin(BasePlugin):
    """
    Advanced CSV exporter plugin with formatting and filtering options.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="csv-exporter",
            version="1.0.0",
            description="Export data to CSV format with advanced formatting",
            author="WebScraper Team",
            plugin_type=PluginType.EXPORTER,
            tags=["csv", "export", "data"],
            config_schema={
                "delimiter": {"type": "string", "default": ","},
                "quote_char": {"type": "string", "default": '"'},
                "include_headers": {"type": "boolean", "default": True},
                "date_format": {"type": "string", "default": "%Y-%m-%d %H:%M:%S"},
                "encoding": {"type": "string", "default": "utf-8"},
                "filter_columns": {"type": "array", "default": []},
                "sort_by": {"type": "string", "default": ""}
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize the CSV exporter plugin."""
        try:
            self.logger.info("Initializing CSV Exporter Plugin")
            
            # Set default configuration values
            self.delimiter = self.get_config_value("delimiter", ",")
            self.quote_char = self.get_config_value("quote_char", '"')
            self.include_headers = self.get_config_value("include_headers", True)
            self.date_format = self.get_config_value("date_format", "%Y-%m-%d %H:%M:%S")
            self.encoding = self.get_config_value("encoding", "utf-8")
            self.filter_columns = self.get_config_value("filter_columns", [])
            self.sort_by = self.get_config_value("sort_by", "")
            
            self._initialized = True
            self._running = True
            
            self.logger.info("CSV Exporter Plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CSV Exporter Plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> PluginResult:
        """
        Export data to CSV format.
        
        Args:
            input_data: Data to export (list of dictionaries or single dictionary)
            context: Optional execution context
            
        Returns:
            PluginResult with CSV content
        """
        try:
            self.logger.info("Executing CSV export")
            
            # Normalize input data to list of dictionaries
            if isinstance(input_data, dict):
                data = [input_data]
            elif isinstance(input_data, list):
                data = input_data
            else:
                raise ValueError("Input data must be a dictionary or list of dictionaries")
            
            if not data:
                return PluginResult(
                    success=False,
                    error="No data to export"
                )
            
            # Filter columns if specified
            if self.filter_columns:
                filtered_data = []
                for row in data:
                    filtered_row = {col: row.get(col, "") for col in self.filter_columns}
                    filtered_data.append(filtered_row)
                data = filtered_data
            
            # Sort data if specified
            if self.sort_by and self.sort_by in data[0]:
                data = sorted(data, key=lambda x: x.get(self.sort_by, ""))
            
            # Get all unique column names
            all_columns = set()
            for row in data:
                all_columns.update(row.keys())
            
            columns = sorted(list(all_columns))
            
            # Create CSV content
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=columns,
                delimiter=self.delimiter,
                quotechar=self.quote_char,
                quoting=csv.QUOTE_MINIMAL
            )
            
            # Write headers if enabled
            if self.include_headers:
                writer.writeheader()
            
            # Write data rows
            for row in data:
                # Format datetime objects
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        formatted_row[key] = value.strftime(self.date_format)
                    else:
                        formatted_row[key] = value
                
                writer.writerow(formatted_row)
            
            csv_content = output.getvalue()
            output.close()
            
            # Generate metadata
            metadata = {
                "rows_exported": len(data),
                "columns": columns,
                "file_size": len(csv_content.encode(self.encoding)),
                "encoding": self.encoding,
                "delimiter": self.delimiter,
                "export_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"CSV export completed: {len(data)} rows, {len(columns)} columns")
            
            return PluginResult(
                success=True,
                data=csv_content,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return PluginResult(
                success=False,
                error=str(e)
            )
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return ["csv", "tsv"]
    
    def validate_data_structure(self, data: Any) -> bool:
        """Validate that data can be exported to CSV."""
        if isinstance(data, dict):
            return True
        elif isinstance(data, list):
            return all(isinstance(item, dict) for item in data)
        return False
    
    def estimate_file_size(self, data: Any) -> int:
        """Estimate the size of the resulting CSV file."""
        if not self.validate_data_structure(data):
            return 0
        
        if isinstance(data, dict):
            data = [data]
        
        # Rough estimation based on string length
        total_chars = 0
        for row in data:
            for value in row.values():
                total_chars += len(str(value)) + 1  # +1 for delimiter
        
        return total_chars
    
    async def export_to_file(self, input_data: Any, filename: str, context: Optional[Dict[str, Any]] = None) -> PluginResult:
        """
        Export data directly to a CSV file.
        
        Args:
            input_data: Data to export
            filename: Output filename
            context: Optional execution context
            
        Returns:
            PluginResult with file information
        """
        try:
            # Get CSV content
            result = await self.execute(input_data, context)
            
            if not result.success:
                return result
            
            # Write to file
            with open(filename, 'w', encoding=self.encoding, newline='') as f:
                f.write(result.data)
            
            # Update metadata
            result.metadata.update({
                "filename": filename,
                "file_written": True
            })
            
            self.logger.info(f"CSV exported to file: {filename}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV to file {filename}: {e}")
            return PluginResult(
                success=False,
                error=str(e)
            )
