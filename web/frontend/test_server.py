#!/usr/bin/env python3
"""
Simple test server to demonstrate the web scraper frontend.
This is a minimal FastAPI server that serves the frontend without all dependencies.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

app = FastAPI(title="Web Scraper Frontend Demo")

# Mount static files
app.mount("/static", StaticFiles(directory="web/frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="web/frontend/templates")

# Mock API endpoints for frontend testing
@app.get("/api/v1/jobs")
async def mock_jobs():
    return {
        "jobs": [
            {
                "id": "job_1",
                "name": "Sample E-commerce Scraping",
                "job_type": "web_scraping",
                "status": "completed",
                "progress": 100,
                "created_at": "2023-12-01T10:00:00Z",
                "priority": "normal"
            },
            {
                "id": "job_2", 
                "name": "News Articles Collection",
                "job_type": "web_scraping",
                "status": "running",
                "progress": 65,
                "created_at": "2023-12-01T11:00:00Z",
                "priority": "high"
            }
        ],
        "total_count": 2
    }

@app.get("/api/v1/agents")
async def mock_agents():
    return {
        "agents": [
            {
                "agent_id": "agent_1",
                "agent_type": "Web Scraping Agent",
                "status": "active",
                "active_tasks": 2,
                "completed_tasks": 15,
                "failed_tasks": 1,
                "last_activity": "2023-12-01T12:00:00Z",
                "capabilities": ["web_scraping", "javascript_rendering", "anti_detection"]
            },
            {
                "agent_id": "agent_2",
                "agent_type": "Data Processing Agent", 
                "status": "idle",
                "active_tasks": 0,
                "completed_tasks": 8,
                "failed_tasks": 0,
                "last_activity": "2023-12-01T11:45:00Z",
                "capabilities": ["data_cleaning", "transformation", "validation"]
            }
        ]
    }

@app.get("/api/v1/monitoring/system")
async def mock_system_metrics():
    return {
        "cpu_usage": 45.2,
        "memory_usage": 62.8,
        "disk_usage": 34.1,
        "uptime": 86400,
        "active_connections": 12,
        "total_requests": 1250
    }

@app.get("/api/v1/jobs/stats/summary")
async def mock_job_statistics():
    return {
        "total_jobs": 156,
        "pending_jobs": 8,
        "running_jobs": 3,
        "completed_jobs": 132,
        "failed_jobs": 13,
        "cancelled_jobs": 0,
        "success_rate": 84.6,
        "average_completion_time": 245.5,
        "jobs_by_type": {
            "web_scraping": 89,
            "data_processing": 45,
            "document_extraction": 22
        },
        "jobs_by_priority": {
            "low": 34,
            "normal": 98,
            "high": 20,
            "urgent": 4
        }
    }

@app.post("/api/v1/jobs")
async def mock_create_job(request: Request):
    return {
        "id": "job_new",
        "name": "New Job",
        "status": "pending",
        "message": "Job created successfully"
    }

# Frontend routes
@app.get("/", response_class=HTMLResponse)
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

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv('WEB_HOST', 'localhost')
    port = int(os.getenv('WEB_PORT', '8000'))
    debug = os.getenv('WEB_DEBUG', 'false').lower() == 'true'

    print("Starting Web Scraper Frontend Demo Server...")
    print(f"Dashboard: http://{host}:{port}/app")
    print(f"Jobs: http://{host}:{port}/app/jobs")
    print(f"Agents: http://{host}:{port}/app/agents")
    print(f"Monitoring: http://{host}:{port}/app/monitoring")
    print(f"Data: http://{host}:{port}/app/data")
    print("\nFrontend Features Implemented:")
    print("   - Responsive Bootstrap 5 UI")
    print("   - Real-time WebSocket support (mock)")
    print("   - Interactive job creation")
    print("   - Agent management")
    print("   - System monitoring with charts")
    print("   - Data export functionality")
    print("   - Mobile-friendly design")
    print("   - Real data integration with fallbacks")

    uvicorn.run(app, host=host, port=port, reload=debug)
