"""
Data Storage agent for the web scraping system.
"""
import asyncio
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import re
import sqlite3
import aiosqlite
import aiofiles
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import shutil
import gzip
import pickle
import csv
import yaml
import xml.etree.ElementTree as ET

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage
from models.task import Task, TaskStatus, TaskType


class DataStorageAgent(Agent):
    """
    Agent responsible for data storage and retrieval.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new data storage agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="data_storage", coordinator_id=coordinator_id)

        # Initialize storage paths
        self.base_path = Path("data")
        self.base_path.mkdir(exist_ok=True)

        # Initialize database connection
        self.db_path = self.base_path / "storage.db"
        self.db_connection = None

        # Cache for query results
        self.query_cache: Dict[str, Dict[str, Any]] = {}

        # Supported formats for data storage
        self.supported_formats = ["json", "csv", "excel", "sqlite", "yaml", "xml", "pickle"]

        # Register message handlers
        self.register_handler("store_data", self._handle_store_data)
        self.register_handler("retrieve_data", self._handle_retrieve_data)
        self.register_handler("query_data", self._handle_query_data)
        self.register_handler("delete_data", self._handle_delete_data)
        self.register_handler("backup_data", self._handle_backup_data)
        self.register_handler("aggregate_results", self._handle_aggregate_results)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the data storage agent."""
        asyncio.create_task(self._periodic_cache_cleanup())
        asyncio.create_task(self._periodic_backup())

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up the query cache."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}", exc_info=True)

    async def _periodic_backup(self) -> None:
        """Periodically backup the database."""
        while self.running:
            try:
                await asyncio.sleep(86400)  # Every day
                if not self.running:
                    break

                await self._backup_database()
            except Exception as e:
                self.logger.error(f"Error in database backup: {str(e)}", exc_info=True)

    def _cleanup_cache(self) -> None:
        """Clean up the query cache."""
        current_time = time.time()
        self.query_cache = {
            key: value
            for key, value in self.query_cache.items()
            if current_time - value.get("timestamp", 0) < 86400  # 24 hours
        }

    async def _backup_database(self) -> None:
        """Backup the database."""
        try:
            # Create backup directory
            backup_dir = self.base_path / "backups"
            backup_dir.mkdir(exist_ok=True)

            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"storage_{timestamp}.db"

            # Copy database file
            shutil.copy2(self.db_path, backup_path)

            # Compress backup
            with open(backup_path, "rb") as f_in:
                with gzip.open(f"{backup_path}.gz", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove uncompressed backup
            backup_path.unlink()

            # Clean up old backups
            self._cleanup_old_backups(backup_dir)

        except Exception as e:
            self.logger.error(f"Error backing up database: {str(e)}", exc_info=True)

    def _cleanup_old_backups(self, backup_dir: Path, max_backups: int = 7) -> None:
        """Clean up old backup files."""
        try:
            # Get all backup files
            backup_files = sorted(
                backup_dir.glob("storage_*.db.gz"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Remove old backups
            for backup_file in backup_files[max_backups:]:
                backup_file.unlink()

        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {str(e)}", exc_info=True)

    async def _handle_store_data(self, message: Message) -> None:
        """
        Handle a data storage request.

        Args:
            message: The message containing data storage parameters.
        """
        if not hasattr(message, "data"):
            self.logger.warning("Received store_data message without data")
            return

        try:
            result = await self.store_data(
                message.data,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error storing data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="data_storage_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_retrieve_data(self, message: Message) -> None:
        """
        Handle a data retrieval request.

        Args:
            message: The message containing data retrieval parameters.
        """
        if not hasattr(message, "data_id"):
            self.logger.warning("Received retrieve_data message without data_id")
            return

        try:
            result = await self.retrieve_data(
                message.data_id,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="data_retrieval_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_query_data(self, message: Message) -> None:
        """
        Handle a data query request.

        Args:
            message: The message containing query parameters.
        """
        if not hasattr(message, "query"):
            self.logger.warning("Received query_data message without query")
            return

        try:
            result = await self.query_data(
                message.query,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error querying data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="data_query_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_delete_data(self, message: Message) -> None:
        """
        Handle a data deletion request.

        Args:
            message: The message containing deletion parameters.
        """
        if not hasattr(message, "data_id"):
            self.logger.warning("Received delete_data message without data_id")
            return

        try:
            result = await self.delete_data(
                message.data_id,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error deleting data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="data_deletion_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_backup_data(self, message: Message) -> None:
        """
        Handle a data backup request.

        Args:
            message: The message containing backup parameters.
        """
        try:
            result = await self.backup_data(
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error backing up data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="data_backup_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def store_data(self, data: Any, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Store data in the database.

        Args:
            data: The data to store.
            options: Storage options.

        Returns:
            A dictionary containing storage results.
        """
        # Generate data ID
        data_id = options.get("data_id") or self._generate_data_id(data)

        # Determine storage format
        format_type = options.get("format", "json")

        # Store data
        if format_type == "json":
            await self._store_json(data, data_id, options)
        elif format_type == "csv":
            await self._store_csv(data, data_id, options)
        elif format_type == "yaml":
            await self._store_yaml(data, data_id, options)
        elif format_type == "xml":
            await self._store_xml(data, data_id, options)
        elif format_type == "pickle":
            await self._store_pickle(data, data_id, options)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Store metadata
        metadata = {
            "data_id": data_id,
            "format": format_type,
            "timestamp": datetime.now().isoformat(),
            "options": options
        }

        await self._store_metadata(metadata)

        return {
            "data_id": data_id,
            "format": format_type,
            "metadata": metadata
        }

    async def retrieve_data(self, data_id: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Retrieve data from the database.

        Args:
            data_id: The ID of the data to retrieve.
            options: Retrieval options.

        Returns:
            A dictionary containing the retrieved data.
        """
        # Retrieve metadata
        metadata = await self._retrieve_metadata(data_id)
        if not metadata:
            raise ValueError(f"Data not found: {data_id}")

        # Determine format
        format_type = metadata["format"]

        # Retrieve data
        if format_type == "json":
            data = await self._retrieve_json(data_id, options)
        elif format_type == "csv":
            data = await self._retrieve_csv(data_id, options)
        elif format_type == "yaml":
            data = await self._retrieve_yaml(data_id, options)
        elif format_type == "xml":
            data = await self._retrieve_xml(data_id, options)
        elif format_type == "pickle":
            data = await self._retrieve_pickle(data_id, options)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return {
            "data": data,
            "metadata": metadata
        }

    async def query_data(self, query: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Query data from the database.

        Args:
            query: The query to execute.
            options: Query options.

        Returns:
            A dictionary containing query results.
        """
        # Check cache
        cache_key = hashlib.md5((query + json.dumps(options, sort_keys=True)).encode()).hexdigest()
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]["result"]

        # Execute query
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]

        # Convert to list of dictionaries
        results = [dict(zip(columns, row)) for row in rows]

        # Store in cache
        self.query_cache[cache_key] = {
            "result": results,
            "timestamp": time.time()
        }

        return results

    async def delete_data(self, data_id: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Delete data from the database.

        Args:
            data_id: The ID of the data to delete.
            options: Deletion options.

        Returns:
            A dictionary containing deletion results.
        """
        # Retrieve metadata
        metadata = await self._retrieve_metadata(data_id)
        if not metadata:
            raise ValueError(f"Data not found: {data_id}")

        # Delete data file
        data_path = self.base_path / f"{data_id}.{metadata['format']}"
        if data_path.exists():
            data_path.unlink()

        # Delete metadata
        await self._delete_metadata(data_id)

        return {
            "data_id": data_id,
            "deleted": True
        }

    async def backup_data(self, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Backup the database.

        Args:
            options: Backup options.

        Returns:
            A dictionary containing backup results.
        """
        await self._backup_database()

        return {
            "backup_completed": True,
            "timestamp": datetime.now().isoformat()
        }

    async def _handle_aggregate_results(self, message: Message) -> None:
        """
        Handle an aggregate results request.

        Args:
            message: The message containing aggregation parameters.
        """
        if not hasattr(message, "chunks"):
            self.logger.warning("Received aggregate_results message without chunks")
            return

        try:
            result = await self.aggregate_results(
                message.chunks,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error aggregating results: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="aggregation_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def aggregate_results(self, chunks: List[Any], options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Aggregate results from multiple sources.

        Args:
            chunks: The data chunks to aggregate.
            options: Aggregation options.

        Returns:
            A dictionary containing the aggregated results.
        """
        from utils.result_aggregator import result_aggregator

        # Get aggregation options
        id_columns = options.get("id_columns", [])
        timestamp_column = options.get("timestamp_column")
        source_column = options.get("source_column")
        source_authority = options.get("source_authority", {})
        similarity_threshold = options.get("similarity_threshold")
        text_columns = options.get("text_columns", [])
        output_format = options.get("output_format", "pandas")

        self.logger.info(f"Aggregating {len(chunks)} result chunks")

        # Use the result aggregator to aggregate the results
        start_time = time.time()
        aggregated_df = result_aggregator.aggregate_results(
            chunks=chunks,
            id_columns=id_columns,
            timestamp_column=timestamp_column,
            source_column=source_column,
            source_authority=source_authority,
            similarity_threshold=similarity_threshold,
            text_columns=text_columns
        )
        end_time = time.time()

        # Convert to the requested output format
        if output_format == "records":
            output_data = aggregated_df.to_dict(orient="records")
        elif output_format == "dict":
            output_data = aggregated_df.to_dict(orient="dict")
        else:  # Default to pandas
            output_data = aggregated_df

        # Store the aggregated results if requested
        if "path" in options:
            path = options["path"]
            format_type = options.get("format", "json").lower()

            # Store data
            store_result = await self.store_data(
                aggregated_df,
                {
                    "format": format_type,
                    "path": path
                }
            )

            return {
                "aggregated_records": len(aggregated_df),
                "execution_time": end_time - start_time,
                "storage_result": store_result,
                "data": output_data
            }

        return {
            "aggregated_records": len(aggregated_df),
            "execution_time": end_time - start_time,
            "data": output_data
        }

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        if task.type == TaskType.STORE_DATA:
            # Convert task parameters to store_data parameters
            data = task.parameters.get("data")
            options = {k: v for k, v in task.parameters.items() if k != "data"}
            return await self.store_data(data, options)
        elif task.type == TaskType.RETRIEVE_DATA:
            # Convert task parameters to retrieve_data parameters
            data_id = task.parameters.get("data_id")
            options = {k: v for k, v in task.parameters.items() if k != "data_id"}
            return await self.retrieve_data(data_id, options)
        elif task.type == TaskType.QUERY_DATA:
            # Convert task parameters to query_data parameters
            query = task.parameters.get("query")
            options = {k: v for k, v in task.parameters.items() if k != "query"}
            return await self.query_data(query, options)
        elif task.type == TaskType.DELETE_DATA:
            # Convert task parameters to delete_data parameters
            data_id = task.parameters.get("data_id")
            options = {k: v for k, v in task.parameters.items() if k != "data_id"}
            return await self.delete_data(data_id, options)
        elif task.type == TaskType.BACKUP_DATA:
            # Convert task parameters to backup_data parameters
            options = task.parameters
            return await self.backup_data(options)
        elif task.type == TaskType.AGGREGATE_RESULTS:
            # Convert task parameters to aggregate_results parameters
            chunks = task.parameters.get("chunks")
            options = {k: v for k, v in task.parameters.items() if k != "chunks"}
            return await self.aggregate_results(chunks, options)
        else:
            raise ValueError(f"Unsupported task type for storage agent: {task.type}")

    def _generate_data_id(self, data: Any) -> str:
        """Generate a unique data ID."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    async def _store_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store metadata in the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    data_id TEXT PRIMARY KEY,
                    format TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    options TEXT NOT NULL
                )
            """)

            await db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?)",
                (
                    metadata["data_id"],
                    metadata["format"],
                    metadata["timestamp"],
                    json.dumps(metadata["options"])
                )
            )

            await db.commit()

    async def _retrieve_metadata(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata from the database."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            async with db.execute(
                "SELECT * FROM metadata WHERE data_id = ?",
                (data_id,)
            ) as cursor:
                row = await cursor.fetchone()

                if row:
                    return {
                        "data_id": row["data_id"],
                        "format": row["format"],
                        "timestamp": row["timestamp"],
                        "options": json.loads(row["options"])
                    }

                return None

    async def _delete_metadata(self, data_id: str) -> None:
        """Delete metadata from the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM metadata WHERE data_id = ?",
                (data_id,)
            )
            await db.commit()

    async def _store_json(self, data: Any, data_id: str, options: Dict[str, Any]) -> None:
        """Store data in JSON format."""
        data_path = self.base_path / f"{data_id}.json"
        async with aiofiles.open(data_path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    async def _store_csv(self, data: Any, data_id: str, options: Dict[str, Any]) -> None:
        """Store data in CSV format."""
        data_path = self.base_path / f"{data_id}.csv"

        if isinstance(data, pd.DataFrame):
            data.to_csv(data_path, index=False)
        else:
            async with aiofiles.open(data_path, "w", newline="") as f:
                writer = csv.writer(f)
                if isinstance(data, list):
                    writer.writerows(data)
                else:
                    writer.writerow([data])

    async def _store_yaml(self, data: Any, data_id: str, options: Dict[str, Any]) -> None:
        """Store data in YAML format."""
        data_path = self.base_path / f"{data_id}.yaml"
        async with aiofiles.open(data_path, "w") as f:
            await f.write(yaml.dump(data))

    async def _store_xml(self, data: Any, data_id: str, options: Dict[str, Any]) -> None:
        """Store data in XML format."""
        data_path = self.base_path / f"{data_id}.xml"

        if isinstance(data, dict):
            root = ET.Element("data")
            self._dict_to_xml(data, root)
            tree = ET.ElementTree(root)
            tree.write(data_path)
        else:
            raise ValueError("XML storage only supports dictionary data")

    async def _store_pickle(self, data: Any, data_id: str, options: Dict[str, Any]) -> None:
        """Store data in pickle format."""
        data_path = self.base_path / f"{data_id}.pickle"
        async with aiofiles.open(data_path, "wb") as f:
            await f.write(pickle.dumps(data))

    async def _retrieve_json(self, data_id: str, options: Dict[str, Any]) -> Any:
        """Retrieve data from JSON format."""
        data_path = self.base_path / f"{data_id}.json"
        async with aiofiles.open(data_path, "r") as f:
            return json.loads(await f.read())

    async def _retrieve_csv(self, data_id: str, options: Dict[str, Any]) -> Any:
        """Retrieve data from CSV format."""
        data_path = self.base_path / f"{data_id}.csv"
        return pd.read_csv(data_path)

    async def _retrieve_yaml(self, data_id: str, options: Dict[str, Any]) -> Any:
        """Retrieve data from YAML format."""
        data_path = self.base_path / f"{data_id}.yaml"
        async with aiofiles.open(data_path, "r") as f:
            return yaml.safe_load(await f.read())

    async def _retrieve_xml(self, data_id: str, options: Dict[str, Any]) -> Any:
        """Retrieve data from XML format."""
        data_path = self.base_path / f"{data_id}.xml"
        tree = ET.parse(data_path)
        root = tree.getroot()
        return self._xml_to_dict(root)

    async def _retrieve_pickle(self, data_id: str, options: Dict[str, Any]) -> Any:
        """Retrieve data from pickle format."""
        data_path = self.base_path / f"{data_id}.pickle"
        async with aiofiles.open(data_path, "rb") as f:
            return pickle.loads(await f.read())

    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element) -> None:
        """Convert dictionary to XML."""
        for key, value in data.items():
            child = ET.SubElement(parent, key)
            if isinstance(value, dict):
                self._dict_to_xml(value, child)
            else:
                child.text = str(value)

    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML to dictionary."""
        result = {}
        for child in element:
            if len(child) > 0:
                result[child.tag] = self._xml_to_dict(child)
            else:
                result[child.tag] = child.text
        return result