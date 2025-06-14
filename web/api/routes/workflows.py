"""
API routes for workflow management and execution.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from plugins.plugin_manager import PluginManager
from plugins.base_plugin import PluginType
from web.api.dependencies import get_agent_manager


logger = logging.getLogger("workflows_api")
router = APIRouter()

# Global plugin manager instance
plugin_manager = PluginManager()


class WorkflowNode(BaseModel):
    """Workflow node definition."""
    id: int
    type: str
    icon: str = ""
    color: str = "#007bff"
    x: float
    y: float
    width: float = 120
    height: float = 60
    properties: Dict[str, Any] = Field(default_factory=dict)


class WorkflowConnection(BaseModel):
    """Connection between workflow nodes."""
    from_node: int = Field(alias="from")
    to_node: int = Field(alias="to")


class WorkflowDefinition(BaseModel):
    """Complete workflow definition."""
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionRequest(BaseModel):
    """Request to execute a workflow."""
    workflow: WorkflowDefinition
    input_data: Dict[str, Any] = Field(default_factory=dict)
    execution_options: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionResult(BaseModel):
    """Result of workflow execution."""
    execution_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None


# In-memory storage for workflow executions (in production, use a database)
workflow_executions: Dict[str, WorkflowExecutionResult] = {}


@router.get("/templates")
async def get_workflow_templates():
    """Get available workflow templates."""
    templates = {
        "ecommerce": {
            "name": "E-commerce Scraper",
            "description": "Extract product information from online stores",
            "nodes": [
                {
                    "id": 1,
                    "type": "Scraper",
                    "icon": "🕷️",
                    "color": "#007bff",
                    "x": 100,
                    "y": 100,
                    "properties": {
                        "url": "https://shop.example.com",
                        "headers": {"User-Agent": "Mozilla/5.0..."},
                        "timeout": 30
                    }
                },
                {
                    "id": 2,
                    "type": "Parser",
                    "icon": "📝",
                    "color": "#28a745",
                    "x": 300,
                    "y": 100,
                    "properties": {
                        "selectors": {
                            "title": "h1.product-title",
                            "price": ".price",
                            "description": ".product-description",
                            "images": "img.product-image"
                        }
                    }
                },
                {
                    "id": 3,
                    "type": "Storage",
                    "icon": "💾",
                    "color": "#6f42c1",
                    "x": 500,
                    "y": 100,
                    "properties": {
                        "format": "csv",
                        "filename": "products",
                        "fields": ["title", "price", "description"]
                    }
                }
            ],
            "connections": [
                {"from": 1, "to": 2},
                {"from": 2, "to": 3}
            ]
        },
        "news": {
            "name": "News Aggregator",
            "description": "Collect articles from news websites",
            "nodes": [
                {
                    "id": 1,
                    "type": "Scraper",
                    "icon": "🕷️",
                    "color": "#007bff",
                    "x": 100,
                    "y": 100,
                    "properties": {
                        "url": "https://news.example.com",
                        "timeout": 30
                    }
                },
                {
                    "id": 2,
                    "type": "Parser",
                    "icon": "📝",
                    "color": "#28a745",
                    "x": 300,
                    "y": 100,
                    "properties": {
                        "selectors": {
                            "headline": "h2.headline",
                            "content": ".article-body",
                            "author": ".author",
                            "date": ".publish-date"
                        }
                    }
                },
                {
                    "id": 3,
                    "type": "Data Transform",
                    "icon": "🔄",
                    "color": "#20c997",
                    "x": 500,
                    "y": 100,
                    "properties": {
                        "operations": ["clean_text", "extract_entities", "sentiment_analysis"]
                    }
                },
                {
                    "id": 4,
                    "type": "Storage",
                    "icon": "💾",
                    "color": "#6f42c1",
                    "x": 700,
                    "y": 100,
                    "properties": {
                        "format": "json",
                        "filename": "articles"
                    }
                }
            ],
            "connections": [
                {"from": 1, "to": 2},
                {"from": 2, "to": 3},
                {"from": 3, "to": 4}
            ]
        }
    }
    
    return {"templates": templates}


@router.post("/execute")
async def execute_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    agent_manager=Depends(get_agent_manager)
):
    """Execute a workflow."""
    import uuid
    
    execution_id = str(uuid.uuid4())
    
    # Create execution record
    execution_result = WorkflowExecutionResult(
        execution_id=execution_id,
        status="running",
        start_time=datetime.now()
    )
    
    workflow_executions[execution_id] = execution_result
    
    # Start workflow execution in background
    background_tasks.add_task(
        _execute_workflow_background,
        execution_id,
        request.workflow,
        request.input_data,
        request.execution_options,
        agent_manager
    )
    
    return {
        "execution_id": execution_id,
        "status": "started",
        "message": "Workflow execution started"
    }


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get the status of a workflow execution."""
    if execution_id not in workflow_executions:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return workflow_executions[execution_id]


@router.get("/executions")
async def list_executions():
    """List all workflow executions."""
    return {"executions": list(workflow_executions.values())}


@router.post("/validate")
async def validate_workflow(workflow: WorkflowDefinition):
    """Validate a workflow definition."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check for nodes
    if not workflow.nodes:
        validation_result["valid"] = False
        validation_result["errors"].append("Workflow must have at least one node")
        return validation_result
    
    # Check node IDs are unique
    node_ids = [node.id for node in workflow.nodes]
    if len(node_ids) != len(set(node_ids)):
        validation_result["valid"] = False
        validation_result["errors"].append("Node IDs must be unique")
    
    # Check connections reference valid nodes
    for connection in workflow.connections:
        if connection.from_node not in node_ids:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Connection references invalid from_node: {connection.from_node}")
        
        if connection.to_node not in node_ids:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Connection references invalid to_node: {connection.to_node}")
    
    # Check for disconnected nodes
    connected_nodes = set()
    for connection in workflow.connections:
        connected_nodes.add(connection.from_node)
        connected_nodes.add(connection.to_node)
    
    disconnected_nodes = set(node_ids) - connected_nodes
    if len(disconnected_nodes) > 1:  # Allow one starting node
        validation_result["warnings"].append(f"Disconnected nodes found: {list(disconnected_nodes)}")
    
    return validation_result


@router.get("/node-types")
async def get_available_node_types():
    """Get available node types for workflow building."""
    node_types = [
        {
            "name": "Scraper",
            "icon": "🕷️",
            "color": "#007bff",
            "description": "Fetch web pages and content",
            "properties": {
                "url": {"type": "string", "required": True},
                "headers": {"type": "object", "required": False},
                "timeout": {"type": "number", "default": 30}
            }
        },
        {
            "name": "Parser",
            "icon": "📝",
            "color": "#28a745",
            "description": "Extract data using CSS selectors",
            "properties": {
                "selectors": {"type": "object", "required": True},
                "format": {"type": "string", "default": "json"}
            }
        },
        {
            "name": "Storage",
            "icon": "💾",
            "color": "#6f42c1",
            "description": "Save extracted data",
            "properties": {
                "format": {"type": "string", "default": "json"},
                "filename": {"type": "string", "required": True}
            }
        },
        {
            "name": "JavaScript",
            "icon": "⚡",
            "color": "#ffc107",
            "description": "Execute JavaScript in browser",
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "timeout": {"type": "number", "default": 10}
            }
        },
        {
            "name": "Authentication",
            "icon": "🔐",
            "color": "#dc3545",
            "description": "Handle login and authentication",
            "properties": {
                "type": {"type": "string", "default": "none"},
                "credentials": {"type": "object", "required": False}
            }
        },
        {
            "name": "Anti-Detection",
            "icon": "🥷",
            "color": "#fd7e14",
            "description": "Avoid detection and blocking",
            "properties": {
                "enabled": {"type": "boolean", "default": True},
                "delay": {"type": "number", "default": 1}
            }
        },
        {
            "name": "Data Transform",
            "icon": "🔄",
            "color": "#20c997",
            "description": "Clean and transform data",
            "properties": {
                "operations": {"type": "array", "default": []}
            }
        },
        {
            "name": "Quality Check",
            "icon": "✅",
            "color": "#17a2b8",
            "description": "Validate data quality",
            "properties": {
                "rules": {"type": "array", "default": []}
            }
        }
    ]
    
    return {"node_types": node_types}


async def _execute_workflow_background(
    execution_id: str,
    workflow: WorkflowDefinition,
    input_data: Dict[str, Any],
    execution_options: Dict[str, Any],
    agent_manager
):
    """Execute workflow in background."""
    execution_result = workflow_executions[execution_id]
    
    try:
        logger.info(f"Starting workflow execution: {execution_id}")
        
        # Simple workflow execution simulation
        # In a real implementation, this would:
        # 1. Build execution graph from nodes and connections
        # 2. Execute nodes in correct order
        # 3. Pass data between connected nodes
        # 4. Handle errors and retries
        
        results = {}
        
        # Sort nodes by execution order (simplified)
        sorted_nodes = sorted(workflow.nodes, key=lambda n: n.x)
        
        for node in sorted_nodes:
            logger.info(f"Executing node: {node.type} (ID: {node.id})")
            
            # Simulate node execution
            await asyncio.sleep(1)  # Simulate processing time
            
            node_result = {
                "node_id": node.id,
                "type": node.type,
                "status": "completed",
                "output": f"Simulated output from {node.type} node"
            }
            
            results[f"node_{node.id}"] = node_result
        
        # Update execution result
        execution_result.status = "completed"
        execution_result.end_time = datetime.now()
        execution_result.results = results
        execution_result.execution_time = (
            execution_result.end_time - execution_result.start_time
        ).total_seconds()
        
        logger.info(f"Workflow execution completed: {execution_id}")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {execution_id} - {e}")
        
        execution_result.status = "failed"
        execution_result.end_time = datetime.now()
        execution_result.errors = [str(e)]
        execution_result.execution_time = (
            execution_result.end_time - execution_result.start_time
        ).total_seconds()


@router.delete("/executions/{execution_id}")
async def cancel_execution(execution_id: str):
    """Cancel a running workflow execution."""
    if execution_id not in workflow_executions:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = workflow_executions[execution_id]
    
    if execution.status == "running":
        execution.status = "cancelled"
        execution.end_time = datetime.now()
        execution.execution_time = (
            execution.end_time - execution.start_time
        ).total_seconds()
        
        return {"message": "Execution cancelled"}
    else:
        raise HTTPException(status_code=400, detail="Execution is not running")


@router.post("/preview")
async def preview_workflow_url(url: str):
    """Preview a URL for workflow building (handles CORS)."""
    try:
        import aiohttp
        from bs4 import BeautifulSoup
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html_content = await response.text()
        
        # Return simplified HTML for preview
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts and styles for security
        for script in soup(["script", "style"]):
            script.decompose()
        
        return {
            "html": str(soup),
            "title": soup.title.string if soup.title else "No title",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to preview URL {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load URL: {str(e)}")
