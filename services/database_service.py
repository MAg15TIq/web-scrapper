"""
Database service for persistent data storage using PostgreSQL.
Provides ORM functionality and data persistence for the enhanced system.
"""
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from contextlib import asynccontextmanager

try:
    import asyncpg
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, JSON
    from sqlalchemy.dialects.postgresql import UUID
    import uuid
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database libraries not available. Install asyncpg, sqlalchemy for persistence.")

from config.langchain_config import get_config
from models.langchain_models import ScrapingRequest, ExecutionPlan, QualityReport


if DATABASE_AVAILABLE:
    Base = declarative_base()
    
    class WorkflowRecord(Base):
        """Database model for workflow records."""
        __tablename__ = "workflows"
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        workflow_id = Column(String(100), unique=True, nullable=False)
        user_input = Column(Text, nullable=False)
        status = Column(String(50), nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        completed_at = Column(DateTime, nullable=True)
        
        # Workflow data (stored as JSON)
        scraping_request = Column(JSON, nullable=True)
        execution_plan = Column(JSON, nullable=True)
        extracted_data = Column(JSON, nullable=True)
        quality_report = Column(JSON, nullable=True)
        final_output = Column(JSON, nullable=True)
        
        # Performance metrics
        execution_time_seconds = Column(Float, nullable=True)
        success_rate = Column(Float, nullable=True)
        data_quality_score = Column(Float, nullable=True)
        
        # Error tracking
        error_count = Column(Integer, default=0)
        errors = Column(JSON, nullable=True)
        recovery_actions = Column(JSON, nullable=True)
    
    class AgentPerformanceRecord(Base):
        """Database model for agent performance tracking."""
        __tablename__ = "agent_performance"
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        agent_id = Column(String(100), nullable=False)
        agent_type = Column(String(50), nullable=False)
        workflow_id = Column(String(100), nullable=False)
        
        # Performance metrics
        tasks_completed = Column(Integer, default=0)
        tasks_failed = Column(Integer, default=0)
        average_execution_time = Column(Float, default=0.0)
        memory_usage_mb = Column(Float, default=0.0)
        cpu_usage_percent = Column(Float, default=0.0)
        success_rate = Column(Float, default=0.0)
        
        # Timestamps
        recorded_at = Column(DateTime, default=datetime.utcnow)
        last_activity = Column(DateTime, default=datetime.utcnow)
    
    class ScrapedDataRecord(Base):
        """Database model for scraped data storage."""
        __tablename__ = "scraped_data"
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        workflow_id = Column(String(100), nullable=False)
        source_url = Column(Text, nullable=False)
        data_type = Column(String(50), nullable=False)
        
        # Data content
        raw_data = Column(JSON, nullable=False)
        processed_data = Column(JSON, nullable=True)
        
        # Quality metrics
        quality_score = Column(Float, nullable=True)
        validation_status = Column(String(20), default="pending")
        
        # Metadata
        scraped_at = Column(DateTime, default=datetime.utcnow)
        processing_time_ms = Column(Integer, nullable=True)
        agent_id = Column(String(100), nullable=True)


class DatabaseService:
    """
    Database service for persistent data storage and retrieval.
    Provides ORM functionality and data persistence for workflows and results.
    """
    
    def __init__(self):
        """Initialize database service."""
        self.logger = logging.getLogger("database.service")
        self.config = get_config().database
        
        # Database connections
        self.engine = None
        self.session_factory = None
        self.connected = False
        
        if not DATABASE_AVAILABLE:
            self.logger.warning("Database libraries not available - using mock implementation")
            self._use_mock = True
            self._mock_storage = {
                "workflows": {},
                "agent_performance": {},
                "scraped_data": {}
            }
        else:
            self._use_mock = False
    
    async def connect(self) -> bool:
        """Connect to PostgreSQL database."""
        if self._use_mock:
            self.logger.info("Using mock database implementation")
            return True
        
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.config.postgres_url,
                pool_size=self.config.postgres_pool_size,
                max_overflow=self.config.postgres_max_overflow,
                pool_timeout=self.config.postgres_pool_timeout,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.connected = True
            self.logger.info(f"Connected to PostgreSQL at {self.config.postgres_host}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            # Fall back to mock implementation
            self._use_mock = True
            self._mock_storage = {
                "workflows": {},
                "agent_performance": {},
                "scraped_data": {}
            }
            return False
    
    async def disconnect(self):
        """Disconnect from database."""
        if self._use_mock:
            return
        
        try:
            if self.engine:
                await self.engine.dispose()
            self.connected = False
            self.logger.info("Disconnected from database")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from database: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager."""
        if self._use_mock:
            yield None  # Mock sessions don't need real session objects
            return
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def save_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Save workflow data to database."""
        try:
            if self._use_mock:
                workflow_id = workflow_data.get("workflow_id")
                self._mock_storage["workflows"][workflow_id] = {
                    **workflow_data,
                    "saved_at": datetime.now()
                }
                self.logger.debug(f"Mock saved workflow: {workflow_id}")
                return True
            
            async with self.get_session() as session:
                # Check if workflow exists
                existing = await session.get(WorkflowRecord, workflow_data.get("workflow_id"))
                
                if existing:
                    # Update existing workflow
                    for key, value in workflow_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new workflow record
                    workflow_record = WorkflowRecord(**workflow_data)
                    session.add(workflow_record)
                
                await session.commit()
                self.logger.debug(f"Saved workflow: {workflow_data.get('workflow_id')}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save workflow: {e}")
            return False
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow data from database."""
        try:
            if self._use_mock:
                return self._mock_storage["workflows"].get(workflow_id)
            
            async with self.get_session() as session:
                workflow = await session.get(WorkflowRecord, workflow_id)
                if workflow:
                    return {
                        "id": str(workflow.id),
                        "workflow_id": workflow.workflow_id,
                        "user_input": workflow.user_input,
                        "status": workflow.status,
                        "created_at": workflow.created_at,
                        "updated_at": workflow.updated_at,
                        "completed_at": workflow.completed_at,
                        "scraping_request": workflow.scraping_request,
                        "execution_plan": workflow.execution_plan,
                        "extracted_data": workflow.extracted_data,
                        "quality_report": workflow.quality_report,
                        "final_output": workflow.final_output,
                        "execution_time_seconds": workflow.execution_time_seconds,
                        "success_rate": workflow.success_rate,
                        "data_quality_score": workflow.data_quality_score,
                        "error_count": workflow.error_count,
                        "errors": workflow.errors,
                        "recovery_actions": workflow.recovery_actions
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow {workflow_id}: {e}")
            return None
    
    async def save_agent_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Save agent performance metrics."""
        try:
            if self._use_mock:
                agent_id = performance_data.get("agent_id")
                workflow_id = performance_data.get("workflow_id")
                key = f"{agent_id}_{workflow_id}"
                self._mock_storage["agent_performance"][key] = {
                    **performance_data,
                    "saved_at": datetime.now()
                }
                return True
            
            async with self.get_session() as session:
                performance_record = AgentPerformanceRecord(**performance_data)
                session.add(performance_record)
                await session.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save agent performance: {e}")
            return False
    
    async def save_scraped_data(self, scraped_data: Dict[str, Any]) -> bool:
        """Save scraped data to database."""
        try:
            if self._use_mock:
                data_id = str(uuid.uuid4())
                self._mock_storage["scraped_data"][data_id] = {
                    **scraped_data,
                    "id": data_id,
                    "saved_at": datetime.now()
                }
                return True
            
            async with self.get_session() as session:
                data_record = ScrapedDataRecord(**scraped_data)
                session.add(data_record)
                await session.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save scraped data: {e}")
            return False
    
    async def get_workflow_history(
        self,
        limit: int = 50,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get workflow history with optional filtering."""
        try:
            if self._use_mock:
                workflows = list(self._mock_storage["workflows"].values())
                if status_filter:
                    workflows = [w for w in workflows if w.get("status") == status_filter]
                return workflows[:limit]
            
            async with self.get_session() as session:
                query = session.query(WorkflowRecord)
                if status_filter:
                    query = query.filter(WorkflowRecord.status == status_filter)
                
                workflows = await query.order_by(WorkflowRecord.created_at.desc()).limit(limit).all()
                
                return [
                    {
                        "workflow_id": w.workflow_id,
                        "status": w.status,
                        "created_at": w.created_at,
                        "execution_time_seconds": w.execution_time_seconds,
                        "success_rate": w.success_rate,
                        "data_quality_score": w.data_quality_score
                    }
                    for w in workflows
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow history: {e}")
            return []
    
    async def get_performance_analytics(
        self,
        agent_type: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get performance analytics for agents."""
        try:
            if self._use_mock:
                # Mock analytics
                return {
                    "total_tasks": 100,
                    "success_rate": 0.95,
                    "average_execution_time": 2.5,
                    "agent_count": 5,
                    "mock_data": True
                }
            
            # Real analytics would query the database
            # This is a simplified version
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "agent_count": 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance analytics: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database service statistics."""
        if self._use_mock:
            return {
                "connected": False,
                "mock_mode": True,
                "workflows_count": len(self._mock_storage["workflows"]),
                "performance_records": len(self._mock_storage["agent_performance"]),
                "scraped_data_records": len(self._mock_storage["scraped_data"])
            }
        else:
            return {
                "connected": self.connected,
                "mock_mode": False,
                "engine_pool_size": self.config.postgres_pool_size if self.engine else 0
            }


# Global database service instance
_database_service: Optional[DatabaseService] = None


async def get_database_service() -> DatabaseService:
    """Get the global database service instance."""
    global _database_service
    
    if _database_service is None:
        _database_service = DatabaseService()
        await _database_service.connect()
    
    return _database_service


async def cleanup_database_service():
    """Cleanup the global database service."""
    global _database_service
    
    if _database_service:
        await _database_service.disconnect()
        _database_service = None
