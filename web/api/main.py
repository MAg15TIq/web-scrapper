"""
FastAPI Web Application for Multi-Agent Web Scraping System
Main application entry point with middleware and route configuration.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import route modules
from web.api.routes import agents, jobs, monitoring, auth, websockets, streaming, security, workflows, plugins
from web.api.middleware.auth import AuthMiddleware
from web.api.middleware.rate_limiting import RateLimitMiddleware
from web.api.dependencies import get_agent_manager, get_database
from web.dashboard.agent_monitor import AgentMonitor
from web.scheduler.job_manager import JobManager
from config.web_config import get_web_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_api")


# Global managers
agent_monitor: AgentMonitor = None
job_manager: JobManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent_monitor, job_manager
    
    # Startup
    logger.info("Starting Multi-Agent Web Scraping API")
    
    try:
        # Initialize configuration
        config = get_web_config()
        
        # Initialize agent monitor
        agent_monitor = AgentMonitor()
        await agent_monitor.start()
        
        # Initialize job manager
        job_manager = JobManager()
        await job_manager.start()
        
        # Store in app state
        app.state.agent_monitor = agent_monitor
        app.state.job_manager = job_manager
        app.state.config = config

        # Start real agent monitoring
        try:
            from web.api.services.agent_manager import agent_manager
            await agent_manager.start_monitoring()
            app.state.real_agent_manager = agent_manager
            logger.info("Real agent monitoring started")
        except Exception as e:
            logger.error(f"Failed to start real agent monitoring: {e}")

        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Agent Web Scraping API")
    
    try:
        if agent_monitor:
            await agent_monitor.stop()
        
        if job_manager:
            await job_manager.stop()

        # Stop real agent monitoring
        if hasattr(app.state, 'real_agent_manager'):
            try:
                await app.state.real_agent_manager.stop_monitoring()
                logger.info("Real agent monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping real agent monitoring: {e}")

        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Web Scraping System API",
    description="Advanced web scraping system with AI-powered agents",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "server_error"
            }
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "2.0.0",
        "services": {
            "agent_monitor": agent_monitor is not None and agent_monitor.is_running,
            "job_manager": job_manager is not None and job_manager.is_running,
            "database": True  # Add actual database check
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Agent Web Scraping System API",
        "version": "2.0.0",
        "description": "Advanced web scraping system with AI-powered agents",
        "docs_url": "/api/docs",
        "health_url": "/health",
        "endpoints": {
            "agents": "/api/v1/agents",
            "jobs": "/api/v1/jobs",
            "monitoring": "/api/v1/monitoring",
            "auth": "/api/v1/auth",
            "websocket": "/ws",
            "streaming": "/api/v1/streaming",
            "security": "/api/v1/security"
        }
    }


# Include API routes
app.include_router(
    agents.router,
    prefix="/api/v1/agents",
    tags=["agents"],
    dependencies=[Depends(get_agent_manager)]
)

app.include_router(
    jobs.router,
    prefix="/api/v1/jobs",
    tags=["jobs"],
    dependencies=[Depends(get_agent_manager)]
)

app.include_router(
    monitoring.router,
    prefix="/api/v1/monitoring",
    tags=["monitoring"]
)

app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["authentication"]
)

app.include_router(
    websockets.router,
    prefix="/ws",
    tags=["websockets"]
)

app.include_router(
    streaming.router,
    prefix="/api/v1/streaming",
    tags=["streaming"]
)

app.include_router(
    security.router,
    prefix="/api/v1",
    tags=["security"]
)

app.include_router(
    workflows.router,
    prefix="/api/v1/workflows",
    tags=["workflows"],
    dependencies=[Depends(get_agent_manager)]
)

app.include_router(
    plugins.router,
    prefix="/api/v1/plugins",
    tags=["plugins"],
    dependencies=[Depends(get_agent_manager)]
)


# Static files and templates
app.mount("/static", StaticFiles(directory="web/frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="web/frontend/templates")

# Authentication routes
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page."""
    return templates.TemplateResponse("login.html", {"request": request})

# Serve frontend pages
@app.get("/app", response_class=HTMLResponse)
@app.get("/app/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Serve main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/app/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    """Serve jobs management page."""
    return templates.TemplateResponse("jobs.html", {"request": request})

@app.get("/app/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Serve agents management page."""
    return templates.TemplateResponse("agents.html", {"request": request})

@app.get("/app/monitoring", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    """Serve monitoring page."""
    return templates.TemplateResponse("monitoring.html", {"request": request})

@app.get("/app/data", response_class=HTMLResponse)
async def data_page(request: Request):
    """Serve data management page."""
    return templates.TemplateResponse("data.html", {"request": request})

@app.get("/app/security", response_class=HTMLResponse)
async def security_page(request: Request):
    """Serve security & compliance page."""
    return templates.TemplateResponse("security.html", {"request": request})

@app.get("/app/workflow-builder", response_class=HTMLResponse)
async def workflow_builder_page(request: Request):
    """Serve visual workflow builder page."""
    return templates.TemplateResponse("workflow-builder.html", {"request": request})


# WebSocket endpoint for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket, client_id: str):
    """WebSocket endpoint for real-time communication."""
    await websockets.handle_websocket_connection(websocket, client_id)


# API documentation customization
@app.get("/api/v1/openapi-custom.json")
async def get_custom_openapi():
    """Get customized OpenAPI schema."""
    from fastapi.openapi.utils import get_openapi
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Multi-Agent Web Scraping System API",
        version="2.0.0",
        description="""
        ## Advanced Web Scraping System with AI-Powered Agents
        
        This API provides access to a sophisticated multi-agent web scraping system that uses:
        
        - **LangChain Integration**: AI-powered decision making and natural language processing
        - **Pydantic AI**: Structured data validation and type safety
        - **Specialized Agents**: 5 core agents for different scraping tasks
        - **Real-time Monitoring**: Live progress tracking and system metrics
        - **Job Scheduling**: Automated and recurring scraping tasks
        
        ### Available Agents
        
        1. **Orchestrator Agent**: Coordinates all scraping operations
        2. **Web Scraping Agent**: Handles HTTP requests and content fetching
        3. **Document Processing Agent**: Processes PDFs, documents, and files
        4. **Data Transformation Agent**: Cleans and transforms extracted data
        5. **Data Output Agent**: Manages data export and storage
        
        ### Key Features
        
        - Natural language command processing
        - Anti-detection and stealth capabilities
        - JavaScript rendering support
        - Multiple output formats (JSON, CSV, Excel, etc.)
        - Real-time progress monitoring
        - Comprehensive error handling and recovery
        """,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "agents",
            "description": "Agent management and configuration"
        },
        {
            "name": "jobs",
            "description": "Job creation, monitoring, and management"
        },
        {
            "name": "monitoring",
            "description": "System monitoring and metrics"
        },
        {
            "name": "authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "websockets",
            "description": "Real-time communication via WebSockets"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Development server configuration
if __name__ == "__main__":
    config = get_web_config()
    
    uvicorn.run(
        "web.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug",
        access_log=True
    )
