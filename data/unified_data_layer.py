"""
Unified Data Layer for Web Scraper System
Provides centralized data persistence and management across all components.
"""
import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Type
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import threading
from contextlib import contextmanager

from config.unified_config import get_unified_config_manager
from models.task import Task, TaskStatus, TaskType
from models.message import Message


class DataSource(str, Enum):
    """Data source types."""
    SQLITE = "sqlite"
    JSON = "json"
    MEMORY = "memory"
    REDIS = "redis"


class EntityType(str, Enum):
    """Entity types in the system."""
    TASK = "task"
    JOB = "job"
    AGENT = "agent"
    SESSION = "session"
    USER = "user"
    RESULT = "result"
    LOG = "log"
    CONFIG = "config"


class DataOperation(str, Enum):
    """Data operations."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SEARCH = "search"


class DataEntity(BaseModel):
    """Base data entity."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_data(self, new_data: Dict[str, Any]) -> None:
        """Update entity data."""
        self.data.update(new_data)
        self.updated_at = datetime.now()


class QueryFilter(BaseModel):
    """Query filter for data operations."""
    entity_type: Optional[EntityType] = None
    field: Optional[str] = None
    value: Optional[Any] = None
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, like
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[str] = None
    order_desc: bool = False


class UnifiedDataLayer:
    """Unified data layer for centralized data management."""
    
    def __init__(self, data_source: DataSource = DataSource.SQLITE):
        """Initialize the unified data layer."""
        self.logger = logging.getLogger("unified_data")
        self.config_manager = get_unified_config_manager()
        self.data_source = data_source
        
        # Data storage paths
        self.data_dir = Path("data")
        self.db_file = self.data_dir / "unified_data.db"
        self.json_dir = self.data_dir / "json"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.json_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache
        self._cache: Dict[str, DataEntity] = {}
        self._cache_enabled = True
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Initialize storage
        self._init_storage()
        
        self.logger.info(f"Unified data layer initialized (source: {data_source})")
    
    def _init_storage(self) -> None:
        """Initialize the storage backend."""
        if self.data_source == DataSource.SQLITE:
            self._init_sqlite()
        elif self.data_source == DataSource.JSON:
            self._init_json()
        elif self.data_source == DataSource.MEMORY:
            self._init_memory()
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Create main entities table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        data TEXT NOT NULL,
                        metadata TEXT NOT NULL
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON entities(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON entities(updated_at)")
                
                # Create tasks table for better performance
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        assigned_to TEXT,
                        priority INTEGER DEFAULT 1,
                        parameters TEXT,
                        result TEXT,
                        error TEXT
                    )
                """)
                
                # Create jobs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        config TEXT,
                        progress INTEGER DEFAULT 0,
                        total_tasks INTEGER DEFAULT 0,
                        completed_tasks INTEGER DEFAULT 0
                    )
                """)
                
                conn.commit()
                
            self.logger.info("SQLite database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    def _init_json(self) -> None:
        """Initialize JSON file storage."""
        for entity_type in EntityType:
            entity_dir = self.json_dir / entity_type.value
            entity_dir.mkdir(exist_ok=True)
        
        self.logger.info("JSON storage initialized")
    
    def _init_memory(self) -> None:
        """Initialize in-memory storage."""
        self._memory_store: Dict[EntityType, Dict[str, DataEntity]] = {}
        for entity_type in EntityType:
            self._memory_store[entity_type] = {}
        
        self.logger.info("Memory storage initialized")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection context manager."""
        if self.data_source != DataSource.SQLITE:
            yield None
            return
        
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_entity(self, entity_type: EntityType, data: Dict[str, Any], 
                     entity_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> DataEntity:
        """Create a new data entity."""
        with self._lock:
            entity = DataEntity(
                id=entity_id or str(uuid.uuid4()),
                entity_type=entity_type,
                data=data,
                metadata=metadata or {}
            )
            
            self._store_entity(entity)
            
            # Update cache
            if self._cache_enabled:
                self._cache[entity.id] = entity
                self._cache_timestamps[entity.id] = datetime.now()
            
            self.logger.debug(f"Created entity: {entity_type.value} ({entity.id})")
            return entity
    
    def get_entity(self, entity_id: str) -> Optional[DataEntity]:
        """Get an entity by ID."""
        with self._lock:
            # Check cache first
            if self._cache_enabled and entity_id in self._cache:
                cache_time = self._cache_timestamps.get(entity_id)
                if cache_time and (datetime.now() - cache_time).seconds < self._cache_ttl:
                    return self._cache[entity_id]
            
            # Load from storage
            entity = self._load_entity(entity_id)
            
            # Update cache
            if entity and self._cache_enabled:
                self._cache[entity_id] = entity
                self._cache_timestamps[entity_id] = datetime.now()
            
            return entity
    
    def update_entity(self, entity_id: str, data: Dict[str, Any], 
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[DataEntity]:
        """Update an existing entity."""
        with self._lock:
            entity = self.get_entity(entity_id)
            if not entity:
                return None
            
            entity.update_data(data)
            if metadata:
                entity.metadata.update(metadata)
            
            self._store_entity(entity)
            
            # Update cache
            if self._cache_enabled:
                self._cache[entity_id] = entity
                self._cache_timestamps[entity_id] = datetime.now()
            
            self.logger.debug(f"Updated entity: {entity.entity_type.value} ({entity_id})")
            return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        with self._lock:
            success = self._delete_entity(entity_id)
            
            # Remove from cache
            if entity_id in self._cache:
                del self._cache[entity_id]
            if entity_id in self._cache_timestamps:
                del self._cache_timestamps[entity_id]
            
            if success:
                self.logger.debug(f"Deleted entity: {entity_id}")
            
            return success
    
    def list_entities(self, entity_type: EntityType, filters: Optional[List[QueryFilter]] = None,
                     limit: Optional[int] = None, offset: Optional[int] = None) -> List[DataEntity]:
        """List entities with optional filtering."""
        with self._lock:
            return self._query_entities(entity_type, filters, limit, offset)
    
    def search_entities(self, query: str, entity_types: Optional[List[EntityType]] = None) -> List[DataEntity]:
        """Search entities by text query."""
        with self._lock:
            return self._search_entities(query, entity_types)
    
    def count_entities(self, entity_type: EntityType, filters: Optional[List[QueryFilter]] = None) -> int:
        """Count entities matching criteria."""
        with self._lock:
            return self._count_entities(entity_type, filters)

    def _store_entity(self, entity: DataEntity) -> None:
        """Store entity in the backend."""
        if self.data_source == DataSource.SQLITE:
            self._store_entity_sqlite(entity)
        elif self.data_source == DataSource.JSON:
            self._store_entity_json(entity)
        elif self.data_source == DataSource.MEMORY:
            self._store_entity_memory(entity)

    def _load_entity(self, entity_id: str) -> Optional[DataEntity]:
        """Load entity from the backend."""
        if self.data_source == DataSource.SQLITE:
            return self._load_entity_sqlite(entity_id)
        elif self.data_source == DataSource.JSON:
            return self._load_entity_json(entity_id)
        elif self.data_source == DataSource.MEMORY:
            return self._load_entity_memory(entity_id)
        return None

    def _delete_entity(self, entity_id: str) -> bool:
        """Delete entity from the backend."""
        if self.data_source == DataSource.SQLITE:
            return self._delete_entity_sqlite(entity_id)
        elif self.data_source == DataSource.JSON:
            return self._delete_entity_json(entity_id)
        elif self.data_source == DataSource.MEMORY:
            return self._delete_entity_memory(entity_id)
        return False

    def _store_entity_sqlite(self, entity: DataEntity) -> None:
        """Store entity in SQLite."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # Store in main entities table
                cursor.execute("""
                    INSERT OR REPLACE INTO entities
                    (id, entity_type, created_at, updated_at, data, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entity.id,
                    entity.entity_type.value,
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    json.dumps(entity.data),
                    json.dumps(entity.metadata)
                ))

                # Store in specialized tables if applicable
                if entity.entity_type == EntityType.TASK:
                    self._store_task_sqlite(entity, cursor)
                elif entity.entity_type == EntityType.JOB:
                    self._store_job_sqlite(entity, cursor)

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to store entity in SQLite: {e}")
            raise

    def _store_task_sqlite(self, entity: DataEntity, cursor) -> None:
        """Store task in specialized table."""
        task_data = entity.data
        cursor.execute("""
            INSERT OR REPLACE INTO tasks
            (id, task_type, status, created_at, updated_at, assigned_to, priority, parameters, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id,
            task_data.get('type', ''),
            task_data.get('status', ''),
            entity.created_at.isoformat(),
            entity.updated_at.isoformat(),
            task_data.get('assigned_to'),
            task_data.get('priority', 1),
            json.dumps(task_data.get('parameters', {})),
            json.dumps(task_data.get('result')),
            json.dumps(task_data.get('error'))
        ))

    def _store_job_sqlite(self, entity: DataEntity, cursor) -> None:
        """Store job in specialized table."""
        job_data = entity.data
        cursor.execute("""
            INSERT OR REPLACE INTO jobs
            (id, name, job_type, status, created_at, updated_at, config, progress, total_tasks, completed_tasks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity.id,
            job_data.get('name', ''),
            job_data.get('job_type', ''),
            job_data.get('status', ''),
            entity.created_at.isoformat(),
            entity.updated_at.isoformat(),
            json.dumps(job_data.get('config', {})),
            job_data.get('progress', 0),
            job_data.get('total_tasks', 0),
            job_data.get('completed_tasks', 0)
        ))

    def _load_entity_sqlite(self, entity_id: str) -> Optional[DataEntity]:
        """Load entity from SQLite."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, entity_type, created_at, updated_at, data, metadata
                    FROM entities WHERE id = ?
                """, (entity_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                return DataEntity(
                    id=row['id'],
                    entity_type=EntityType(row['entity_type']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    data=json.loads(row['data']),
                    metadata=json.loads(row['metadata'])
                )

        except Exception as e:
            self.logger.error(f"Failed to load entity from SQLite: {e}")
            return None

    def _delete_entity_sqlite(self, entity_id: str) -> bool:
        """Delete entity from SQLite."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # Delete from main table
                cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))

                # Delete from specialized tables
                cursor.execute("DELETE FROM tasks WHERE id = ?", (entity_id,))
                cursor.execute("DELETE FROM jobs WHERE id = ?", (entity_id,))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            self.logger.error(f"Failed to delete entity from SQLite: {e}")
            return False

    def _store_entity_json(self, entity: DataEntity) -> None:
        """Store entity in JSON file."""
        try:
            entity_file = self.json_dir / entity.entity_type.value / f"{entity.id}.json"
            entity_file.parent.mkdir(exist_ok=True)

            with open(entity_file, 'w') as f:
                json.dump(entity.model_dump(), f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to store entity in JSON: {e}")
            raise

    def _load_entity_json(self, entity_id: str) -> Optional[DataEntity]:
        """Load entity from JSON file."""
        try:
            # Search in all entity type directories
            for entity_type in EntityType:
                entity_file = self.json_dir / entity_type.value / f"{entity_id}.json"
                if entity_file.exists():
                    with open(entity_file, 'r') as f:
                        data = json.load(f)
                        # Convert datetime strings back to datetime objects
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                        return DataEntity(**data)

            return None

        except Exception as e:
            self.logger.error(f"Failed to load entity from JSON: {e}")
            return None

    def _delete_entity_json(self, entity_id: str) -> bool:
        """Delete entity from JSON file."""
        try:
            # Search in all entity type directories
            for entity_type in EntityType:
                entity_file = self.json_dir / entity_type.value / f"{entity_id}.json"
                if entity_file.exists():
                    entity_file.unlink()
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to delete entity from JSON: {e}")
            return False

    def _store_entity_memory(self, entity: DataEntity) -> None:
        """Store entity in memory."""
        self._memory_store[entity.entity_type][entity.id] = entity

    def _load_entity_memory(self, entity_id: str) -> Optional[DataEntity]:
        """Load entity from memory."""
        for entity_type in EntityType:
            if entity_id in self._memory_store[entity_type]:
                return self._memory_store[entity_type][entity_id]
        return None

    def _delete_entity_memory(self, entity_id: str) -> bool:
        """Delete entity from memory."""
        for entity_type in EntityType:
            if entity_id in self._memory_store[entity_type]:
                del self._memory_store[entity_type][entity_id]
                return True
        return False

    def _query_entities(self, entity_type: EntityType, filters: Optional[List[QueryFilter]] = None,
                       limit: Optional[int] = None, offset: Optional[int] = None) -> List[DataEntity]:
        """Query entities with filters."""
        if self.data_source == DataSource.SQLITE:
            return self._query_entities_sqlite(entity_type, filters, limit, offset)
        elif self.data_source == DataSource.JSON:
            return self._query_entities_json(entity_type, filters, limit, offset)
        elif self.data_source == DataSource.MEMORY:
            return self._query_entities_memory(entity_type, filters, limit, offset)
        return []

    def _query_entities_sqlite(self, entity_type: EntityType, filters: Optional[List[QueryFilter]] = None,
                              limit: Optional[int] = None, offset: Optional[int] = None) -> List[DataEntity]:
        """Query entities from SQLite."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT id, entity_type, created_at, updated_at, data, metadata FROM entities WHERE entity_type = ?"
                params = [entity_type.value]

                # Add filters
                if filters:
                    for filter_obj in filters:
                        if filter_obj.field and filter_obj.value is not None:
                            if filter_obj.operator == "eq":
                                query += f" AND json_extract(data, '$.{filter_obj.field}') = ?"
                                params.append(filter_obj.value)
                            elif filter_obj.operator == "like":
                                query += f" AND json_extract(data, '$.{filter_obj.field}') LIKE ?"
                                params.append(f"%{filter_obj.value}%")

                # Add ordering
                query += " ORDER BY created_at DESC"

                # Add limit and offset
                if limit:
                    query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                entities = []
                for row in rows:
                    entity = DataEntity(
                        id=row['id'],
                        entity_type=EntityType(row['entity_type']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        data=json.loads(row['data']),
                        metadata=json.loads(row['metadata'])
                    )
                    entities.append(entity)

                return entities

        except Exception as e:
            self.logger.error(f"Failed to query entities from SQLite: {e}")
            return []

    def _query_entities_memory(self, entity_type: EntityType, filters: Optional[List[QueryFilter]] = None,
                              limit: Optional[int] = None, offset: Optional[int] = None) -> List[DataEntity]:
        """Query entities from memory."""
        entities = list(self._memory_store[entity_type].values())

        # Apply filters
        if filters:
            for filter_obj in filters:
                if filter_obj.field and filter_obj.value is not None:
                    entities = [e for e in entities if self._apply_filter(e, filter_obj)]

        # Sort by created_at desc
        entities.sort(key=lambda x: x.created_at, reverse=True)

        # Apply offset and limit
        if offset:
            entities = entities[offset:]
        if limit:
            entities = entities[:limit]

        return entities

    def _apply_filter(self, entity: DataEntity, filter_obj: QueryFilter) -> bool:
        """Apply a filter to an entity."""
        try:
            field_value = entity.data.get(filter_obj.field)

            if filter_obj.operator == "eq":
                return field_value == filter_obj.value
            elif filter_obj.operator == "ne":
                return field_value != filter_obj.value
            elif filter_obj.operator == "like":
                return str(filter_obj.value).lower() in str(field_value).lower()
            elif filter_obj.operator == "in":
                return field_value in filter_obj.value

            return True

        except Exception:
            return False

    def _count_entities(self, entity_type: EntityType, filters: Optional[List[QueryFilter]] = None) -> int:
        """Count entities matching criteria."""
        if self.data_source == DataSource.SQLITE:
            return self._count_entities_sqlite(entity_type, filters)
        elif self.data_source == DataSource.MEMORY:
            return len(self._query_entities_memory(entity_type, filters))
        return 0

    def _count_entities_sqlite(self, entity_type: EntityType, filters: Optional[List[QueryFilter]] = None) -> int:
        """Count entities in SQLite."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT COUNT(*) FROM entities WHERE entity_type = ?"
                params = [entity_type.value]

                # Add filters
                if filters:
                    for filter_obj in filters:
                        if filter_obj.field and filter_obj.value is not None:
                            if filter_obj.operator == "eq":
                                query += f" AND json_extract(data, '$.{filter_obj.field}') = ?"
                                params.append(filter_obj.value)

                cursor.execute(query, params)
                return cursor.fetchone()[0]

        except Exception as e:
            self.logger.error(f"Failed to count entities in SQLite: {e}")
            return 0

    def _search_entities(self, query: str, entity_types: Optional[List[EntityType]] = None) -> List[DataEntity]:
        """Search entities by text query."""
        # Simple text search implementation
        results = []
        search_types = entity_types or list(EntityType)

        for entity_type in search_types:
            entities = self.list_entities(entity_type)
            for entity in entities:
                # Search in data fields
                entity_text = json.dumps(entity.data).lower()
                if query.lower() in entity_text:
                    results.append(entity)

        return results

    def clear_cache(self) -> None:
        """Clear the entity cache."""
        with self._lock:
            self._cache.clear()
            self._cache_timestamps.clear()
            self.logger.info("Entity cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get data layer statistics."""
        stats = {
            "data_source": self.data_source.value,
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
            "entity_counts": {}
        }

        for entity_type in EntityType:
            count = self.count_entities(entity_type)
            stats["entity_counts"][entity_type.value] = count

        return stats


# Global unified data layer instance
_unified_data_layer: Optional[UnifiedDataLayer] = None


def get_unified_data_layer() -> UnifiedDataLayer:
    """Get the global unified data layer instance."""
    global _unified_data_layer
    if _unified_data_layer is None:
        _unified_data_layer = UnifiedDataLayer()
    return _unified_data_layer
